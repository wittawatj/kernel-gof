"""
Module containing many types of goodness-of-fit test methods.
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import kgof.data as data
import kgof.util as util
import kgof.kernel as kernel

import scipy.stats as stats
import tensorflow as tf

class GofTest(object):
    """Abstract class for a goodness-of-fit test."""
    __metaclass__ = ABCMeta

    def __init__(self, alpha):
        """
        alpha: significance level of the test
        """
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, dat):
        """perform the goodness-of-fit test and return values computed in a dictionary:
        {
            alpha: 0.01, 
            pvalue: 0.0002, 
            test_stat: 2.3, 
            h0_rejected: True, 
            time_secs: ...
        }

        dat: an instance of Data
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, dat):
        """Compute the test statistic"""
        raise NotImplementedError()


#------------------------------------------------------

class FSSD(GofTest):
    """
    Goodness-of-fit test using The Finite Set Stein Discrepancy statistic.
    and a set of paired test locations. The statistic is n*FSSD^2. 
    The statistic can be negative because of the unbiased estimator.

    H0: the sample follows p
    H1: the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, k, V, alpha=0.01, n_simulate=1000, seed=10):
        """
        p: an instance of UnnormalizedDensity
        k: a Kernel object
        V: J x dx numpy array of J locations to test the difference
        alpha: significance level 
        n_simulate: The number of times to simulate from the null distribution
            (weighted sum of chi-squares). Must be a positive integer.
        """
        super(FSSD, self).__init__(alpha)
        self.p = p
        self.k = k
        self.V = V 
        self.n_simulate = n_simulate
        self.seed = seed

    def perform_test(self, dat, return_simulated_stats=False):
        with util.ContextTimer() as t:
            alpha = self.alpha
            n_simulate = self.n_simulate
            X = dat.data()
            n = X.shape[0]
            J = self.V.shape[0]

            nfssd, fea_tensor = self.compute_stat(dat, return_feature_tensor=True)
            # n x d*J
            Tau = fea_tensor.reshape(n, -1)
            cov = np.cov(Tau.T)

            arr_nfssd, eigs = FSSD.list_simulate_spectral(cov, J, n_simulate,
                    seed=self.seed)
            # approximate p-value with the permutations 
            pvalue = np.mean(arr_nfssd > nfssd)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': nfssd,
                'h0_rejected': pvalue < alpha, 'n_simulate': n_simulate,
                'time_secs': t.secs, 
                }
        if return_simulated_stats:
            results['sim_stats'] = arr_nfssd
        return results

    def compute_stat(self, dat, return_feature_tensor=False):
        """
        The statistic is n*FSSD^2.
        """
        X = dat.data()
        n = X.shape[0]

        # n x d x J
        Xi = self.feature_tensor(X)
        # n x d*J
        Tau = Xi.reshape(n, -1)
        t1 = np.sum(np.sum(Tau/np.sqrt(n-1), 0)**2 )
        t2 = np.sum( (Tau/np.sqrt(n-1))**2 )
        stat = t1 - t2

        #print 'Xi: {0}'.format(Xi)
        #print 'Tau: {0}'.format(Tau)
        #print 't1: {0}'.format(t1)
        #print 't2: {0}'.format(t2)
        #print 'stat: {0}'.format(stat)
        if return_feature_tensor:
            return stat, Xi
        else:
            return stat

    def feature_tensor(self, X):
        """
        Compute the feature tensor which is n x d x J.
        The feature tensor can be used to compute the statistic, and the
        covariance matrix for simulating from the null distribution.

        X: n x d data numpy array

        return an n x d x J numpy array
        """
        k = self.k
        # n x d matrix of gradients
        grad_logp = self.p.grad_log(X)
        # n x J matrix
        K = k.eval(X, self.V)

        list_grads = [k.grad_x(X, v) for v in self.V]
        stack0 = np.concatenate([each[np.newaxis, :] for each in list_grads], axis=0)
        #a numpy array G of size n x d x J such that G[:, :, J]
        #    is the derivative of k(X, V_j) with respect to X.
        dKdV = np.transpose(stack0, (1, 2, 0))

        # n x d x J tensor
        grad_logp_K = util.outer_rows(grad_logp, K)
        Xi = grad_logp_K + dKdV
        return Xi

    @staticmethod 
    def list_simulate_spectral(cov, J, n_simulate=1000, seed=82):
        """
        Simulate the null distribution using the spectrums of the covariance
        matrix.  This is intended to be used to approximate the null
        distribution.

        - features_x: n x Dx where Dx is the number of features for x 
        - features_y: n x Dy

        Return (a numpy array of simulated n*FSSD values, eigenvalues of cov)
        """
        # eigen decompose 
        eigs, _ = np.linalg.eig(cov)
        eigs = np.real(eigs)
        # sort in decreasing order 
        eigs = -np.sort(-eigs)
        sim_fssds = FSSD.simulate_null_dist(eigs, J, n_simulate=n_simulate,
                seed=seed)
        return sim_fssds, eigs

    @staticmethod 
    def simulate_null_dist(eigs, J, n_simulate=1000, seed=7):
        """
        Simulate the null distribution using the spectrums of the covariance 
        matrix of the U-statistic. The simulated statistic is n*FSSD^2 where
        FSSD is an unbiased estimator.

        - eigs: a numpy array of estimated eigenvalues of the covariance
          matrix. eigs is of length d*J, where d is the input dimension, and 
        - J: the number of test locations.

        Return a numpy array of simulated statistics.
        """
        d = len(eigs)/J
        assert d>0
        # draw at most d x J x block_size values at a time
        block_size = max(20, int(1000.0/(d*J)))
        fssds = np.zeros(n_simulate)
        from_ind = 0
        with util.NumpySeedContext(seed=seed):
            while from_ind < n_simulate:
                to_draw = min(block_size, n_simulate-from_ind)
                # draw chi^2 random variables. 
                chi2 = np.random.randn(d*J, to_draw)**2

                # an array of length to_draw 
                sim_fssds = eigs.dot(chi2-1.0)
                # store 
                end_ind = from_ind+to_draw
                fssds[from_ind:end_ind] = sim_fssds
                from_ind = end_ind
        return fssds

