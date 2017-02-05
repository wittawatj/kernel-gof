"""
Module containing many types of goodness-of-fit test methods.
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import autograd
import autograd.numpy as np
import kgof.data as data
import kgof.util as util
import kgof.kernel as kernel
import logging
import matplotlib.pyplot as plt

import scipy
import scipy.stats as stats

class GofTest(object):
    """Abstract class for a goodness-of-fit test.
    Many subclasses will likely take in a Density object in the constructor,
    representing a fixed distribution to compare against.
    """
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
        k: a DifferentiableKernel object
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
            # Make sure it is a matrix i.e, np.cov returns a scalar when Tau is
            # 1d.
            #cov = np.cov(Tau.T) + np.zeros((1, 1))
            cov = Tau.T.dot(Tau/n)

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
        stat = FSSD.ustat_h1_mean_variance(Xi, return_variance=False)

        #print 'Xi: {0}'.format(Xi)
        #print 'Tau: {0}'.format(Tau)
        #print 't1: {0}'.format(t1)
        #print 't2: {0}'.format(t2)
        #print 'stat: {0}'.format(stat)
        if return_feature_tensor:
            return stat, Xi
        else:
            return stat

    def get_H1_mean_variance(self, dat):
        """
        Return the mean and variance under H1 of the test statistic (divided by
        n).
        """
        X = dat.data()
        Xi = self.feature_tensor(X)
        mean, variance = FSSD.ustat_h1_mean_variance(Xi, return_variance=True)
        return mean, variance

    def feature_tensor(self, X):
        """
        Compute the feature tensor which is n x d x J.
        The feature tensor can be used to compute the statistic, and the
        covariance matrix for simulating from the null distribution.

        X: n x d data numpy array

        return an n x d x J numpy array
        """
        k = self.k
        n, d = X.shape
        # n x d matrix of gradients
        grad_logp = self.p.grad_log(X)
        #assert np.all(util.is_real_num(grad_logp))
        # n x J matrix
        #print 'V'
        #print self.V
        K = k.eval(X, self.V)
        #assert np.all(util.is_real_num(K))

        list_grads = np.array([np.reshape(k.gradX_y(X, v), (1, n, d)) for v in self.V])
        stack0 = np.concatenate(list_grads, axis=0)
        #a numpy array G of size n x d x J such that G[:, :, J]
        #    is the derivative of k(X, V_j) with respect to X.
        dKdV = np.transpose(stack0, (1, 2, 0))

        # n x d x J tensor
        grad_logp_K = util.outer_rows(grad_logp, K)
        #print 'grad_logp'
        #print grad_logp.dtype
        #print grad_logp
        #print 'K'
        #print K
        Xi = grad_logp_K + dKdV
        return Xi

    @staticmethod
    def power_criterion(p, data, k, test_locs, reg=1e-2):
        """
        Compute the mean and standard deviation of the statistic under H1.
        Return mean/sd.
        """
        X = data.data()
        V = test_locs
        fssd = FSSD(p, k, V)
        fea_tensor = fssd.feature_tensor(X)
        u_mean, u_variance = FSSD.ustat_h1_mean_variance(fea_tensor,
                return_variance=True)

        # mean/sd criterion 
        obj = u_mean/np.sqrt(u_variance + reg) 
        return obj

    @staticmethod
    def ustat_h1_mean_variance(fea_tensor, return_variance=True):
        """
        Compute the mean and variance of the asymptotic normal distribution 
        under H1 of the test statistic.

        fea_tensor: feature tensor obtained from feature_tensor()
        return_variance: If false, avoid computing and returning the variance.

        Return the mean [and the variance]
        """
        Xi = fea_tensor
        #print 'Xi'
        #print Xi
        #assert np.all(util.is_real_num(Xi))
        n = Xi.shape[0]
        assert n>1, 'Need n > 1 to compute the mean of the statistic.'
        # n x d*J
        Tau = Xi.reshape(n, -1)
        t1 = np.sum(np.sum(Tau/np.sqrt(n-1), 0)**2 )
        t2 = np.sum( (Tau/np.sqrt(n-1))**2 )
        # stat is the mean
        stat = t1 - t2

        if not return_variance:
            return stat

        # compute the variance 
        # mu: d*J vector
        mu = np.mean(Tau, 0)
        variance = 4*np.mean(np.dot(Tau, mu)**2) + 4*np.sum(mu**2)**2
        return stat, variance

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

    @staticmethod
    def fssd_grid_search_kernel(p, dat, test_locs, list_kernel):
        """
        Linear search for the best kernel in the list that maximizes 
        the test power criterion, fixing the test locations to V.

        - p: UnnormalizedDensity
        - dat: a Data object
        - list_kernel: list of kernel candidates 

        return: (best kernel index, array of test power criteria)
        """
        V = test_locs
        X = dat.data()
        n_cand = len(list_kernel)
        objs = np.zeros(n_cand)
        for i in xrange(n_cand):
            ki = list_kernel[i]
            obj = FSSD.power_criterion(p, dat, ki, test_locs)
            logging.info('(%d), obj: %5.4g, k: %s' %(i, obj, str(ki)))

        #Widths that come early in the list 
        # are preferred if test powers are equal.
        #bestij = np.unravel_index(objs.argmax(), objs.shape)
        besti = objs.argmax()
        return besti, objs

# end of FSSD
# --------------------------------------
class GaussFSSD(FSSD):
    """
    FSSD using an isotropic Gaussian kernel.
    """
    def __init__(self, p, sigma2, V, alpha=0.01, n_simulate=1000, seed=10):
        k = kernel.KGauss(sigma2)
        super(GaussFSSD, self).__init__(p, k, V, alpha, n_simulate, seed)

    @staticmethod 
    def power_criterion(p, dat, gwidth, test_locs, reg=1e-4):
        k = kernel.KGauss(gwidth)
        return FSSD.power_criterion(p, dat, k, test_locs, reg)

    @staticmethod
    def optimize_auto_init(p, dat, J, **ops):
        """
        Optimize parameters by calling optimize_locs_widths(). Automatically 
        initialize the test locations and the Gaussian width.
        """
        assert J>0
        # Use grid search to initialize the gwidth
        X = dat.data()
        n_gwidth_cand = 5
        gwidth_factors = 2.0**np.linspace(-3, 3, n_gwidth_cand) 
        med2 = util.meddistance(X, 1000)**2

        k = kernel.KGauss(med2*2)
        # fit a Gaussian to the data and draw to initialize V0
        V0 = util.fit_gaussian_draw(X, J, seed=829, reg=1e-6)
        list_gwidth = np.hstack( ( (med2)*gwidth_factors ) )
        besti, objs = GaussFSSD.grid_search_gwidth(p, dat, V0, list_gwidth)
        gwidth = list_gwidth[besti]
        assert util.is_real_num(gwidth), 'gwidth not real. Was %s'%str(gwidth)
        assert gwidth > 0, 'gwidth not positive. Was %.3g'%gwidth
        logging.info('After grid search, gwidth=%.3g'%gwidth)

        
        V_opt, gwidth_opt, info = GaussFSSD.optimize_locs_widths(p, dat,
                gwidth, V0, **ops) 

        # set the width bounds
        #fac_min = 5e-2
        #fac_max = 5e3
        #gwidth_lb = fac_min*med2
        #gwidth_ub = fac_max*med2
        #gwidth_opt = max(gwidth_lb, min(gwidth_opt, gwidth_ub))
        return V_opt, gwidth_opt, info

    @staticmethod
    def grid_search_gwidth(p, dat, test_locs, list_gwidth):
        """
        Linear search for the best Gaussian width in the list that maximizes 
        the test power criterion, fixing the test locations. 

        - V: a J x dx np-array for J test locations 

        return: (best width index, list of test power objectives)
        """
        list_gauss_kernel = [kernel.KGauss(gw) for gw in list_gwidth]
        besti, objs = FSSD.fssd_grid_search_kernel(p, dat, test_locs,
                list_gauss_kernel)
        return besti, objs

    @staticmethod
    def optimize_locs_widths(p, dat, gwidth0, test_locs0, reg=1e-2,
            max_iter=100,  tol_fun=1e-3, disp=False, locs_bounds_frac=0.5,
            gwidth_lb=None, gwidth_ub=None,
            ):
        """
        Optimize the test locations and the Gaussian kernel width by 
        maximizing a test power criterion. data should not be the same data as
        used in the actual test (i.e., should be a held-out set). 
        This function is deterministic.

        - data: a Data object
        - test_locs0: Jxd numpy array. Initial V.
        - reg: reg to add to the mean/sqrt(variance) criterion to become
            mean/sqrt(variance + reg)
        - gwidth0: initial value of the Gaussian width^2
        - max_iter: #gradient descent iterations
        - optimization_method: a string specifying an optimization method as described 
            by scipy.optimize.minimize  
        - tol_fun: termination tolerance of the objective value
        - disp: True to print convergence messages
        - locs_bounds_frac: When making box bounds for the test_locs, extend
            the box defined by coordinate-wise min-max by std of each coordinate
            multiplied by this number.

        #- If the lb, ub bounds are None, use fraction of the median heuristics 
        #    to automatically set the bounds.
        
        Return (V test_locs, gaussian width, optimization info log)
        """
        J = test_locs0.shape[0]
        X = dat.data()
        n, d = X.shape

        # Parameterize the Gaussian width with its square root (then square later)
        # to automatically enforce the positivity.
        def obj(sqrt_gwidth, V):
            return -GaussFSSD.power_criterion(p, dat, sqrt_gwidth**2, V, reg=reg)
        flatten = lambda gwidth, V: np.hstack((gwidth, V.reshape(-1)))
        def unflatten(x):
            sqrt_gwidth = x[0]
            V = np.reshape(x[1:], (J, d))
            return sqrt_gwidth, V

        def flat_obj(x):
            sqrt_gwidth, V = unflatten(x)
            return obj(sqrt_gwidth, V)
        # gradient
        #grad_obj = autograd.elementwise_grad(flat_obj)
        # Initial point
        x0 = flatten(np.sqrt(gwidth0), test_locs0)
        
        #make sure that the optimized gwidth is not too small or too large.
        fac_min = 1e-2 
        fac_max = 5e2
        med2 = util.meddistance(X, subsample=1000)**2
        if gwidth_lb is None:
            gwidth_lb = max(fac_min*med2, 1e-3)
        if gwidth_ub is None:
            gwidth_ub = min(fac_max*med2, 1e6)

        # Make a box to bound test locations
        X_std = np.std(X, axis=0)
        # X_min: length-d array
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        # V_lb: J x d
        V_lb = np.tile(X_min - locs_bounds_frac*X_std, (J, 1))
        V_ub = np.tile(X_max + locs_bounds_frac*X_std, (J, 1))
        # (J*d+1) x 2. Take square root because we parameterize with the square
        # root
        x0_lb = np.hstack((np.sqrt(gwidth_lb), np.reshape(V_lb, -1)))
        x0_ub = np.hstack((np.sqrt(gwidth_ub), np.reshape(V_ub, -1)))
        x0_bounds = zip(x0_lb, x0_ub)

        # optimize. Time the optimization as well.
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        grad_obj = autograd.elementwise_grad(flat_obj)
        with util.ContextTimer() as timer:
            opt_result = scipy.optimize.minimize(
              flat_obj, x0, method='L-BFGS-B', 
              bounds=x0_bounds,
              tol=tol_fun, 
              options={
                  'maxiter': max_iter, 'ftol': tol_fun, 'disp': disp,
                  'gtol': 1.0e-04,
                  },
              jac=grad_obj,
            )

        opt_result = dict(opt_result)
        opt_result['time_secs'] = timer.secs
        x_opt = opt_result['x']
        sq_gw_opt, V_opt = unflatten(x_opt)
        gw_opt = sq_gw_opt**2

        assert util.is_real_num(gw_opt), 'gw_opt is not real. Was %s' % str(gw_opt)

        return V_opt, gw_opt, opt_result


# ------------------

def bootstrapper_rademacher(n):
    """
    Produce a sequence of i.i.d {-1, 1} random variables.
    Suitable for boostrapping on an i.i.d. sample.
    """
    return 2.0*np.random.randint(0, 1+1, n)-1.0

def bootstrapper_multinomial(n):
    """
    Produce a sequence of i.i.d Multinomial(n; 1/n,... 1/n) random variables.
    This is described on page 5 of Liu et al., 2016 (ICML 2016).
    """
    M = np.random.multinomial(n, np.ones(n)/float(n), size=1) 
    return M.reshape(-1) - 1.0/n

class KernelSteinTest(GofTest):
    """
    Goodness-of-fit test using kernelized Stein discrepancy test of 
    Chwialkowski et al., 2016 and Liu et al., 2016 in ICML 2016.
    Mainly follow the details in Chwialkowski et al., 2016.
    The test statistic is n*V_n where V_n is a V-statistic.

    - This test runs in O(n^2 d^2) time.

    H0: the sample follows p
    H1: the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, k, bootstrapper=bootstrapper_rademacher, alpha=0.01,
            n_simulate=500, seed=11):
        """
        p: an instance of UnnormalizedDensity
        k: a KSTKernel object
        bootstrapper: a function: (n) |-> numpy array of n weights 
            to be multiplied in the double sum of the test statistic for generating 
            boostrap samples from the null distribution.
        alpha: significance level 
        n_simulate: The number of times to simulate from the null distribution
            by bootstrapping. Must be a positive integer.
        """
        super(KernelSteinTest, self).__init__(alpha)
        self.p = p
        self.k = k
        self.bootstrapper = bootstrapper
        self.n_simulate = n_simulate
        self.seed = seed

    def perform_test(self, dat, return_simulated_stats=False, return_ustat_gram=False):
        """
        dat: a instance of Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            n_simulate = self.n_simulate
            X = dat.data()
            n = X.shape[0]

            _, H = self.compute_stat(dat, return_ustat_gram=True)
            test_stat = n*np.mean(H)
            # bootrapping
            sim_stats = np.zeros(n_simulate)
            with util.NumpySeedContext(seed=self.seed):
                for i in range(n_simulate):
                   W = self.bootstrapper(n)
                   boot_stat = W.dot(H.dot(W/float(n)))
                   # This is a bootstrap version of n*V_n
                   sim_stats[i] = boot_stat
 
            # approximate p-value with the permutations 
            pvalue = np.mean(sim_stats > test_stat)
 
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha, 'n_simulate': n_simulate,
                 'time_secs': t.secs, 
                 }
        if return_simulated_stats:
            results['sim_stats'] = sim_stats
        if return_ustat_gram:
            results['H'] = H
            
        return results


    def compute_stat(self, dat, return_ustat_gram=False):
        """
        Compute the V statistic as in Section 2.2 of Chwialkowski et al., 2016.
        return_ustat_gram: If True, then return the n x n matrix used to
            compute the statistic (by taking the mean of all the elements)
        """
        X = dat.data()
        n, d = X.shape
        k = self.k
        # n x d matrix of gradients
        grad_logp = self.p.grad_log(X)
        # n x n
        gram_glogp = grad_logp.dot(grad_logp.T)
        # n x n
        K = k.eval(X, X)

        B = np.zeros((n, n))
        C = np.zeros((n, n))
        for i in range(d):
            grad_logp_i = grad_logp[:, i]
            B += k.gradX_Y(X, X, i)*grad_logp_i
            C += (k.gradY_X(X, X, i).T * grad_logp_i).T

        H = K*gram_glogp + B + C + k.gradXY_sum(X, X)
        # V-statistic
        stat = n*np.mean(H)
        if return_ustat_gram:
            return stat, H
        else:
            return stat

        #print 't1: {0}'.format(t1)
        #print 't2: {0}'.format(t2)
        #print 't3: {0}'.format(t3)
        #print 't4: {0}'.format(t4)

# end KernelSteinTest

class LinearKernelSteinTest(GofTest):
    """
    Goodness-of-fit test using the linear-version of kernelized Stein
    discrepancy test of Liu et al., 2016 in ICML 2016. Described in Liu et al.,
    2016. 
    - This test runs in O(n d^2) time.
    - test stat = sqrt(n_half)*linear-time Stein discrepancy
    - Asymptotically normal under both H0 and H1.

    H0: the sample follows p
    H1: the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, k, alpha=0.01, seed=11):
        """
        p: an instance of UnnormalizedDensity
        k: a LinearKSTKernel object
        alpha: significance level 
        n_simulate: The number of times to simulate from the null distribution
            by bootstrapping. Must be a positive integer.
        """
        super(LinearKernelSteinTest, self).__init__(alpha)
        self.p = p
        self.k = k
        self.seed = seed

    def perform_test(self, dat):
        """
        dat: a instance of Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            X = dat.data()
            n = X.shape[0]

            # H: length-n vector
            _, H = self.compute_stat(dat, return_pointwise_stats=True)
            test_stat = np.sqrt(n)*np.mean(H)
            stat_var = np.mean(H**2) 
            pvalue = stats.norm.sf(test_stat, loc=0, scale=np.sqrt(stat_var) )
 
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha, 'time_secs': t.secs, 
                 }
        return results


    def compute_stat(self, dat, return_pointwise_stats=False):
        """
        Compute the linear-time statistic described in Eq. 17 of Liu et al., 2016
        """
        X = dat.data()
        n, d = X.shape
        k = self.k
        # Divide the sample into two halves of equal size. 
        n_half = n/2
        X1 = X[:n_half, :]
        # May throw away last sample
        X2 = X[n_half:(2*n_half), :]
        assert X1.shape[0] == n_half
        assert X2.shape[0] == n_half
        # score vectors
        S1 = self.p.grad_log(X1)
        # n_half x d
        S2 = self.p.grad_log(X2)
        Kvec = k.pair_eval(X1, X2)

        A = np.sum(S1*S2, 1)*Kvec
        B = np.sum(S2*k.pair_gradX_Y(X1, X2), 1)
        C = np.sum(S1*k.pair_gradY_X(X1, X2), 1)
        D = k.pair_gradXY_sum(X1, X2)

        H = A + B + C + D
        assert len(H) == n_half
        stat = np.mean(H)
        if return_pointwise_stats:
            return stat, H
        else:
            return stat

