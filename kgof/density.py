"""
Module containing implementations of some unnormalized probability density 
functions.
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import autograd
import autograd.numpy as np
import kgof.config as config
import kgof.util as util
import scipy.stats as stats

class UnnormalizedDensity(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def log_den(self, X):
        """
        Evaluate this log of the unnormalized density on the n points in X.

        X: n x d numpy array

        Return a one-dimensional numpy array of length n.
        """
        raise NotImplementedError()

    def grad_log(self, X):
        """
        Evaluate the gradients (with respect to the input) of the log density at
        each of the n points in X.

        X: n x d numpy array.

        Return an n x d numpy array of gradients.
        """
        g = autograd.elementwise_grad(self.log_den)
        G = g(X)
        return G

    @abstractmethod
    def dim(self):
        """
        Return the dimension of the input.
        """
        raise NotImplementedError()

# end UnnormalizedDensity

class IsotropicNormal(UnnormalizedDensity):
    """
    Unnormalized density of an isotropic multivariate normal distribution.
    """
    def __init__(self, mean, variance):
        """
        mean: a numpy array of length d for the mean 
        variance: a positive floating-point number for the variance.
        """
        self.mean = mean 
        self.variance = variance

    def log_den(self, X):
        mean = self.mean 
        variance = self.variance
        unden = -np.sum((X-mean)**2, 1)/(2.0*variance)
        return unden

    def dim(self):
        return len(self.mean)

class Normal(UnnormalizedDensity):
    """
    A multivariate normal distribution.
    """
    def __init__(self, mean, cov):
        """
        mean: a numpy array of length d.
        cov: d x d numpy array for the covariance.
        """
        self.mean = mean 
        self.cov = cov
        assert mean.shape[0] == cov.shape[0]
        assert cov.shape[0] == cov.shape[1]
        E, V = np.linalg.eig(cov)
        if np.any(np.abs(E) <= 1e-7):
            raise ValueError('covariance matrix is not full rank.')
        # The precision matrix
        self.prec = np.dot(np.dot(V, np.diag(1.0/E)), V.T)
        print self.prec

    def log_den(self, X):
        mean = self.mean 
        X0 = X - mean
        X0prec = np.dot(X0, self.prec)
        unden = -np.sum(X0prec*X0, 1)/2.0
        return unden

    def dim(self):
        return len(self.mean)


# end IsotropicNormal



