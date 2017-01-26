
"""Module containing kernel related classes"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import kgof.config as config
import numpy as np
import tensorflow as tf

class Kernel(object):
    """Abstract class for kernels. Inputs to all methods are numpy arrays."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def eval(self, X, Y):
        """Evalute the kernel on data X and Y """
        pass

    #@abstractmethod
    #def pair_eval(self, X, Y):
    #    """Evaluate k(x1, y1), k(x2, y2), ..."""
    #    pass


class TFKernel(object):
    """
    Abstract class for kernels where inputs are TensorFlow variables.
    This is useful, for instance, for automatic differentiation of the kernel 
    with respect to one input argument.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        # tf_X is a TensorFlow variable representing the first input argument 
        # to the kernel.
        self.tf_X = tf.placeholder(config.tensorflow_config['default_float'])
        self.tf_Y = tf.placeholder(config.tensorflow_config['default_float'])

    @abstractmethod
    def tf_eval(self, X, Y):
        """
        X, Y are TensorFlow variables.
        X: n1 x d matrix.
        Y: n2 x d matrix.

        Return a TensorFlow variable representing an n1 x n2 matrix.
        """
        raise NotImplementedError()

    def grad_x(self, X, Y):
        """
        Compute the gradient with respect to X (the first argument of the
        kernel) using TensorFlow. This method calls tf_eval().
        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a numpy array G of size nx x d x ny such that G[:, :, j]
            is the derivative of k(X, Y_j) with respect to X, where Y_j 
            denotes jth row of Y.
        """
        #tf_X = self.tf_X
        #tf_Y = self.tf_Y
        tf_X = tf.placeholder(config.tensorflow_config['default_float'], shape=X.shape)
        tf_Y = tf.placeholder(config.tensorflow_config['default_float'], shape=Y.shape)
        tf_K = self.tf_eval(tf_X, tf_Y)
        tf_K_cols = tf.unstack(tf_K, axis=1)
        tf_Kdxs = [tf.gradients(col, [tf_X])[0] for col in tf_K_cols]

        with tf.Session() as sess:
            # cols_dxs is a list. Each element is an nx x d numpy array. 
            cols_dxs = sess.run(tf_Kdxs, feed_dict={tf_X: X, tf_Y: Y})
        #return np.concatenate([col[np.newaxis, :] for col in cols_dxs], axis=0)
        stack0 = np.concatenate([col[np.newaxis, :] for col in cols_dxs], axis=0)
        return np.transpose(stack0, (1, 2, 0))

# end class TFKernel

class DifferentiableKernel(TFKernel, Kernel):
    """
    A differentiable kernel which is both a Kernel and a TFKernel.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        super(DifferentiableKernel, self).__init__()


class KHoPoly(Kernel):
    """Homogeneous polynomial kernel of the form
    (x.dot(y))**d
    """
    def __init__(self, degree):
        assert degree > 0
        self.degree = degree

    def eval(self, X, Y):
        return X.dot(Y.T)**self.degree

    def pair_eval(self, X, Y):
        return np.sum(X*Y, 1)**self.degree

    def __str__(self):
        return 'KHoPoly(d=%d)'%self.degree


class KLinear(Kernel):
    def eval(self, X, Y):
        return X.dot(Y.T)

    def pair_eval(self, X, Y):
        return np.sum(X*Y, 1)

    def __str__(self):
        return "KLinear()"


class KGauss(DifferentiableKernel):

    def __init__(self, sigma2):
        super(KGauss, self).__init__()
        assert sigma2 > 0, 'sigma2 must be > 0. Was %s'%str(sigma2)
        self.sigma2 = sigma2

    def eval(self, X, Y):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X : n1 x d numpy array
        Y : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        D2 = np.sum(X**2, 1)[:, np.newaxis] - 2*X.dot(Y.T) + np.sum(Y**2, 1)
        K = np.exp(-D2/self.sigma2)
        return K

    def tf_eval(self, X, Y):
        """
        X, Y are TensorFlow variables.
        X: n1 x d matrix.
        Y: n2 x d matrix.

        Return a TensorFlow variable representing an n1 x n2 matrix.
        """
        norm1 = tf.reduce_sum(tf.square(X), 1)
        norm2 = tf.reduce_sum(tf.square(Y), 1)
        col1 = tf.reshape(norm1, [-1, 1])
        row2 = tf.reshape(norm2, [1, -1])
        D2 = col1 - 2*tf.matmul(X, tf.transpose(Y)) + row2
        K = tf.exp(-D2/self.sigma2)
        return K

    def __str__(self):
        return "KGauss(%.3f)"%self.sigma2

    #def pair_eval(self, X, Y):
    #    """
    #    Evaluate k(x1, y1), k(x2, y2), ...

    #    Parameters
    #    ----------
    #    X, Y : n x d numpy array

    #    Return
    #    -------
    #    a numpy array with length n
    #    """
    #    (n1, d1) = X.shape
    #    (n2, d2) = Y.shape
    #    assert n1==n2, 'Two inputs must have the same number of instances'
    #    assert d1==d2, 'Two inputs must have the same dimension'
    #    D2 = np.sum( (X-Y)**2, 1)
    #    Kvec = np.exp(-D2/self.sigma2)
    #    return Kvec



