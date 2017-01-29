
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


class TFKernel(object):
    """
    Abstract class for kernels where inputs are TensorFlow variables.
    This is useful, for instance, for automatic differentiation of the kernel 
    with respect to one input argument.

    It is possible to automatically implement eval() from Kernel class as well.
    However, this can be slow due to the overhead of TensorFlow.
    """
    __metaclass__ = ABCMeta

    # TensorFlow variables for this class. Need to build them only once.
    # A dictionary: variable name |-> TensorFlow variable.
    tf_vars = None
    tf_parameters = None

    @classmethod
    def build_graph(cls):
       if cls.tf_vars is None and cls.tf_parameters is None:
           # This should create the TensorFlow variables.
           tf_params = cls.tf_params()
           cls.tf_parameters = tf_params
           # tf_X is a TensorFlow variable representing the first input argument 
           # to the kernel.
           tf_X = tf.placeholder(config.tensorflow_config['default_float'], name='X')
           tf_Y = tf.placeholder(config.tensorflow_config['default_float'], name='Y')
           tf_y = tf.placeholder(config.tensorflow_config['default_float'], name='y')
           # This node is for computing the Gram matrix.
           tf_kery = cls.tf_eval(tf_X, tf_y, **tf_params)
           # prebuild TensorFlow node for evaluating the gradient of k(X, y)
           # w.r.t X.
           tf_dKdX = tf.gradients(tf_kery, [tf_X])[0]

           tf_vars = {}
           tf_vars['tf_X'] = tf_X
           tf_vars['tf_Y'] = tf_Y
           tf_vars['tf_y'] = tf_y
           tf_vars['tf_kery'] = tf_kery
           tf_vars['tf_dKdX'] = tf_dKdX
           cls.tf_vars = tf_vars

    def __init__(self, **params):
        self.build_graph()
        self.params = params

        self.tf_X = self.tf_vars['tf_X']
        self.tf_Y = self.tf_vars['tf_Y'] 
        self.tf_y = self.tf_vars['tf_y'] 
        self.tf_kery = self.tf_vars['tf_kery']
        self.tf_dKdX = self.tf_vars['tf_dKdX']

    @classmethod
    def tf_params(cls):
        """
        Return a dictionary whose keys are variables names, and values are 
        TensorFlow variables representing parameters of the unnormalized density,
        creating the TensorFlow variables in the process.
        """
        raise NotImplementedError()

    @classmethod
    def tf_eval(self, X, Y, **tf_params):
        """
        X, Y are TensorFlow variables.
        X: n1 x d matrix.
        Y: n2 x d matrix.

        Return a TensorFlow variable representing an n1 x n2 matrix.
        """
        raise NotImplementedError()

# end class TFKernel

class KSTKernel(object):
    """
    Interface specifiying methods a kernel has to implement to be used with 
    the Kernelized Stein discrepancy test of Chwialkowski et al., 2016 and 
    Liu et al., 2016 (ICML 2016 papers).
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def gradX_Y(self, X, Y, dim):
       """
       Compute the gradient with respect to the dimension dim of X in k(X, Y).

       X: nx x d
       Y: ny x d

       Return a numpy array of size nx x ny.
       """
       raise NotImplementedError()

    @abstractmethod
    def gradY_X(self, X, Y, dim):
       """
       Compute the gradient with respect to the dimension dim of Y in k(X, Y).

       X: nx x d
       Y: ny x d

       Return a numpy array of size nx x ny.
       """
       raise NotImplementedError()

    #@abstractmethod
    #def gradX_Y_avgX(self, X, Y):
    #    """
    #    Compute the gradient with respect to X of k(X, Y) then average over all X.

    #    X: nx x d
    #    Y: ny x d

    #    Return a numpy array of size ny x d
    #    """
    #    raise NotImplementedError()

    #@abstractmethod
    #def gradY_X_avgY(self, X, Y):
    #    """
    #    Compute the gradient with respect to Y of k(X, Y) then average over all Y.

    #    X: nx x d
    #    Y: ny x d

    #    Return a numpy array of size nx x d
    #    """
    #    raise NotImplementedError()

    @abstractmethod
    def gradXY_sum(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\partial^2 k(x, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array. 

        Return a nx x ny numpy array of the derivatives.
        """
        raise NotImplementedError()
    

class DifferentiableKernel(TFKernel, Kernel):
    """
    A differentiable kernel which is both a Kernel and a TFKernel.
    """
    __metaclass__ = ABCMeta

    def __init__(self, **params):
        super(DifferentiableKernel, self).__init__(**params)

    def gradX_y(self, X, y):
        """
        Compute the gradient with respect to X (the first argument of the
        kernel) using TensorFlow. This method calls tf_eval().
        X: nx x d numpy array.
        y: numpy array of length d.

        Return a numpy array G of size nx x d, the derivative of k(X, y) with
        respect to X.
        """
        tf_X = self.tf_X
        tf_y = self.tf_y
        tf_dKdX = self.tf_dKdX

        to_feed = {tf_X: X, tf_y: y[np.newaxis, :]}
        #print 'to_feed: {0}'.format(to_feed)
        # Take into account possible parameters of the kernel
        if self.tf_parameters is not None:
            for var_name, tfv in self.tf_parameters.iteritems():
                if var_name not in self.params:
                    raise ValueError('The parameter "{0}" is not specified.'.format(var_name))
                to_feed[tfv] = self.params[var_name]
        with tf.Session() as sess:
            Kdx = sess.run(tf_dKdX, feed_dict=to_feed)
        return Kdx


class KGauss(DifferentiableKernel, KSTKernel):

    def __init__(self, sigma2):
        assert sigma2 > 0, 'sigma2 must be > 0. Was %s'%str(sigma2)
        self.sigma2 = sigma2
        super(KGauss, self).__init__(sigma2=sigma2)

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
        K = np.exp(-D2/(2.0*self.sigma2))
        return K

    @classmethod
    def tf_eval(cls, X, Y, sigma2):
        """
        X, Y, sigma2 are TensorFlow variables.
        X: n1 x d matrix.
        Y: n2 x d matrix.

        Return a TensorFlow variable representing an n1 x n2 matrix.
        """
        norm1 = tf.reduce_sum(tf.square(X), 1)
        norm2 = tf.reduce_sum(tf.square(Y), 1)
        col1 = tf.reshape(norm1, [-1, 1])
        row2 = tf.reshape(norm2, [1, -1])
        D2 = col1 - 2*tf.matmul(X, tf.transpose(Y)) + row2
        K = tf.exp(-D2/(2.0*sigma2))
        return K

    @classmethod
    def tf_params(cls):
        # Remember that keys should match params dictionary in the constructor.
        tf_vars = {'sigma2': tf.placeholder(config.tensorflow_config['default_float'], 
            name='sigma2')}
        return tf_vars


    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).
 
        X: nx x d
        Y: ny x d
 
        Return a numpy array of size nx x ny.
        """
        sigma2 = self.sigma2
        K = self.eval(X, Y)
        Diff = X[:, [dim]] - Y[:, [dim]].T
        G = -K*Diff/sigma2
        return G

    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).
 
        X: nx x d
        Y: ny x d
 
        Return a numpy array of size nx x ny.
        """
        return -self.gradX_Y(X, Y, dim)
 
    #def gradX_Y_avgX(self, X, Y):
    #    """
    #    Compute the gradient with respect to X of k(X, Y) then average over all X.

    #    X: nx x d
    #    Y: ny x d

    #    Return a numpy array of size ny x d
    #    """
    #    assert X.shape[1] == Y.shape[1]
    #    ny, d = Y.shape
    #    sigma2 = self.sigma2
    #    K = self.eval(X, Y)
    #    # resulting gradient
    #    G = np.zeros((ny, d))
    #    for i in range(d):
    #        # Do O(n^2) for d dimensions. Memory cost: O(n^2)
    #        Xi = X[:, [i]]
    #        Yi = Y[:, [i]]
    #        # nx x ny matrix
    #        Diffi = Xi-Yi.T
    #        # ny vector
    #        G[:, i] = -np.mean(K*Diffi, axis=0)/sigma2
    #    return G

    #def gradY_X_avgY(self, X, Y):
    #    """
    #    Compute the gradient with respect to Y of k(X, Y) then average over all Y.

    #    X: nx x d
    #    Y: ny x d

    #    Return a numpy array of size nx x d
    #    """
    #    return -self.gradX_Y_avgX(Y, X)

    def gradXY_sum(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\partial^2 k(x, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array. 

        Return a nx x ny numpy array of the derivatives.
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        d = d1
        sigma2 = self.sigma2
        D2 = np.sum(X**2, 1)[:, np.newaxis] - 2*X.dot(Y.T) + np.sum(Y**2, 1)
        K = np.exp(-D2/(2.0*sigma2))
        G = K/sigma2*(d - D2/sigma2)
        return G

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
    #    Kvec = np.exp(-D2/(2.0*self.sigma2))
    #    return Kvec



