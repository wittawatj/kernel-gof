
"""Module containing kernel related classes"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import autograd
import autograd.numpy as np
import kgof.config as config
#import numpy as np

class Kernel(object):
    """Abstract class for kernels. Inputs to all methods are numpy arrays."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def eval(self, X, Y):
        """Evalute the kernel on data X and Y """
        pass


class KSTKernel(Kernel):
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
    

class DifferentiableKernel(Kernel):
    __metaclass__ = ABCMeta

    @abstractmethod
    def gradX_y(self, X, y):
        """
        Compute the gradient with respect to X (the first argument of the
        kernel) using TensorFlow. This method calls tf_eval().
        X: nx x d numpy array.
        y: numpy array of length d.

        Return a numpy array G of size nx x d, the derivative of k(X, y) with
        respect to X.
        """
        raise NotImplementedError()


class KGauss(DifferentiableKernel, KSTKernel):

    def __init__(self, sigma2):
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
        D2 = np.sum(X**2, 1)[:, np.newaxis] - 2*np.dot(X, Y.T) + np.sum(Y**2, 1)
        K = np.exp(-D2/(2.0*self.sigma2))
        return K

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
        D2 = np.sum(X**2, 1)[:, np.newaxis] - 2*np.dot(X, Y.T) + np.sum(Y**2, 1)
        K = np.exp(-D2/(2.0*sigma2))
        G = K/sigma2*(d - D2/sigma2)
        return G

    def gradX_y(self, X, y):
        f = lambda X: self.eval(X, y[np.newaxis, :])
        g = autograd.elementwise_grad(f)
        G = g(X)
        assert G.shape[0] == X.shape[0]
        assert G.shape[1] == X.shape[1]
        return G

    def __str__(self):
        return "KGauss(%.3f)"%self.sigma2

