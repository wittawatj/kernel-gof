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

# end IsotropicNormal



