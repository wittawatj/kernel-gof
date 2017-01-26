"""
Module containing implementations of some unnormalized probability density 
functions.
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import kgof.config as config
import kgof.util as util
import math
import numpy as np
import scipy.stats as stats
import tensorflow as tf


class UnnormalizedDensity(object):
    """
    An abstract class for an unnormalized differentiable density function.
    Subclasses should call __init__() of this base class.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.tf_X = tf.placeholder(config.tensorflow_config['default_float'])

    @abstractmethod
    def tf_log_den(self, X):
        """
        TensorFlow version of the log_den(.) method.
        This method is intended to be used with TensorFlow to find the
        derivative of the density with respect to the input vector.

        X: n x d TensorFlow variable 

        Return a one-dimensional TensorFlow variable of length n.
        """
        pass

    #@abstractmethod
    #def dim(self):
    #    """Return the expected input dimension of this density."""
    #    pass

    def log_den(self, X):
        """
        Evaluate this log of the unnormalized density on the n points in X.

        This implementation is done by automatically evaluating the TensorFlow
        graph through tf_log_den(). Subclasses may override this if necessary.

        X: n x d numpy array

        Return a one-dimensional numpy array of length n.
        """
        tf_X = self.tf_X
        tf_p = self.tf_log_den(tf_X)

        with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #init = tf.initialize_variables([tf_X])
            #sess.run(init)
            px = sess.run(tf_p, feed_dict={tf_X: X})
        return px

    def grad_log(self, X):
        """
        Evaluate the gradients (with respect to the input) of the log density at
        each of the n points in X.

        This implementation is done by automatically computing the gradient 
        through tf_log_den() using TensorFlow. Subclasses may override this if
        necessary.

        X: n x d numpy array.

        Return an n x d numpy array of gradients.
        """
        tf_X = self.tf_X
        tf_p = self.tf_log_den(tf_X)
        tf_pdx = tf.gradients(tf_p, [tf_X])[0]

        with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            pdx = sess.run(tf_pdx, feed_dict={tf_X: X})
        return pdx

# end of UnnormalizedDensity

class IsotropicNormal(UnnormalizedDensity):
    """
    Unnormalized density of an isotropic multivariate normal distribution.
    """
    def __init__(self, mean, variance):
        """
        mean: one-dimensional numpy array of length d for the mean 
        variance: a positive floating-point number for the variance.
        """
        super(IsotropicNormal, self).__init__()
        self.mean = mean 
        self.variance = variance

    def tf_log_den(self, X):
        # X if a TensorFlow variable 
        d = len(self.mean)
        tf_mean = tf.constant(self.mean, name='mean')
        tf_unden = -tf.reduce_sum(tf.square(X - tf_mean), 1)/self.variance
        return tf_unden

    #def dim(self):
    #   return len(self.mean)

# end of IsotropicNormal
    
class CallableDensity(UnnormalizedDensity):
    """
    A convenient unnormalized density that can be specified by a function or a 
    callable object.
    """
    def __init__(self, func):
        """
        func: a function handle taking a TensorFlow variable, and returning 
            density values.
        """
        self.func = func 

    def tf_log_den(self, X):
        return self.func(X)

# end of CallableDensity

