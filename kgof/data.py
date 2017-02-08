"""
Module containing data structures for representing datasets.
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import math
import autograd.numpy as np
import kgof.util as util
import scipy.stats as stats

class Data(object):
    """
    Class representing a dataset i.e., en encapsulation of a data matrix 
    whose rows are vectors drawn from a distribution.
    """

    def __init__(self, X):
        """
        :param X: n x d numpy array for dataset X
        """
        self.X = X

        if not np.all(np.isfinite(X)):
            print 'X:'
            print util.fullprint(X)
            raise ValueError('Not all elements in X are finite.')

    def __str__(self):
        mean_x = np.mean(self.X, 0)
        std_x = np.std(self.X, 0) 
        prec = 4
        desc = ''
        desc += 'E[x] = %s \n'%(np.array_str(mean_x, precision=prec ) )
        desc += 'Std[x] = %s \n' %(np.array_str(std_x, precision=prec))
        return desc

    def dim(self):
        """Return the dimension of the data."""
        dx = self.X.shape[1]
        return dx

    def sample_size(self):
        return self.X.shape[0]

    def n(self):
        return self.sample_size()

    def data(self):
        """Return the data matrix."""
        return self.X

    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """Split the dataset into training and test sets.         

        Return (Data for tr, Data for te)"""
        X = self.X
        nx, dx = X.shape
        Itr, Ite = util.tr_te_indices(nx, tr_proportion, seed)
        tr_data = Data(X[Itr, :])
        te_data = Data(X[Ite, :])
        return (tr_data, te_data)

    def subsample(self, n, seed=87):
        """Subsample without replacement. Return a new Data. """
        if n > self.X.shape[0]:
            raise ValueError('n should not be larger than sizes of X')
        ind_x = util.subsample_ind( self.X.shape[0], n, seed )
        return Data(self.X[ind_x, :])

    def clone(self):
        """
        Return a new Data object with a separate copy of each internal 
        variable, and with the same content.
        """
        nX = np.copy(self.X)
        return Data(nX)

    def __add__(self, data2):
        """
        Merge the current Data with another one.
        Create a new Data and create a new copy for all internal variables.
        """
        copy = self.clone()
        copy2 = data2.clone()
        nX = np.vstack((copy.X, copy2.X))
        return Data(nX)

### end Data class        


class DataSource(object):
    """
    A source of data allowing resampling. Subclasses may prefix 
    class names with DS. 
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def sample(self, n, seed):
        """Return a Data. Returned result should be deterministic given 
        the input (n, seed)."""
        raise NotImplementedError()

    #@abstractmethod
    #def dim(self):
    #    """Return the dimension of the data. """
    #    raise NotImplementedError()

#  end DataSource

class DSIsotropicNormal(DataSource):
    """
    A DataSource providing samples from a mulivariate isotropic normal
    distribution.
    """
    def __init__(self, mean, variance):
        """
        mean: a numpy array of length d for the mean 
        variance: a positive floating-point number for the variance.
        """
        assert len(mean.shape) == 1
        self.mean = mean 
        self.variance = variance

    def sample(self, n, seed=2):
        with util.NumpySeedContext(seed=seed):
            d = len(self.mean)
            mean = self.mean
            variance = self.variance
            X = np.random.randn(n, d)*np.sqrt(variance) + mean
            return Data(X)

class DSNormal(DataSource):
    """
    A DataSource implementing a multivariate Gaussian.
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

    def sample(self, n, seed=3):
        with util.NumpySeedContext(seed=seed):
            mvn = stats.multivariate_normal(self.mean, self.cov)
            X = mvn.rvs(size=n)
            if len(X.shape) ==1:
                # This can happen if d=1
                X = X[:, np.newaxis]
            return Data(X)

class DSLaplace(DataSource):
    """
    A DataSource for a multivariate Laplace distribution.
    """
    def __init__(self, d, loc=0, scale=1):
        """
        loc: location 
        scale: scale parameter.
        Described in https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.laplace.html#numpy.random.laplace
        """
        assert d > 0
        self.d = d
        self.loc = loc
        self.scale = scale

    def sample(self, n, seed=4):
        with util.NumpySeedContext(seed=seed):
            X = np.random.laplace(loc=self.loc, scale=self.scale, size=(n, self.d))
            return Data(X)

class DSGaussBernRBM(DataSource):
    """
    A DataSource implementing a Gaussian-Bernoulli Restricted Boltzmann Machine.
    The probability of the latent vector h is controlled by the vector c.
    The parameterization of the Gaussian-Bernoulli RBM is given in 
    density.GaussBernRBM.

    - It turns out that this is equivalent to drawing a vector of {-1, 1} for h
        according to h ~ Discrete(sigmoid(2c)).
    - Draw x | h ~ N(B*h+b, I)
    """
    def __init__(self, B, b, c, burnin=50):
        """
        B: a dx x dh matrix 
        b: a numpy array of length dx
        c: a numpy array of length dh
        burnin: burn-in iterations when doring Gibbs sampling
        """
        assert burnin >= 0
        dh = len(c)
        dx = len(b)
        assert B.shape[0] == dx
        assert B.shape[1] == dh
        assert dx > 0
        assert dh > 0
        self.B = B
        self.b = b
        self.c = c
        self.burnin = burnin

    @staticmethod
    def sigmoid(x):
        """
        x: a numpy array.
        """
        return 1.0/(1+np.exp(-x))

    def _blocked_gibbs_next(self, X, H):
        """
        Sample from the mutual conditional distributions.
        """
        dh = H.shape[1]
        n, dx = X.shape
        B = self.B
        b = self.b

        # Draw H.
        XBC = np.dot(X, self.B) + self.c
        # Ph: n x dh matrix
        Ph = DSGaussBernRBM.sigmoid(2*XBC)
        # H: n x dh
        # Don't know how to make this faster.
        H = np.zeros((n, dh))
        for i in range(n):
            for j in range(dh):
                H[i, j] = stats.bernoulli.rvs(p=Ph[i, j], size=1)*2 - 1.0
        assert np.all(np.abs(H) - 1 <= 1e-6 )
        # Draw X.
        # mean: n x dx
        mean = np.dot(H, B.T) + b
        X = np.random.randn(n, dx) + mean
        return X, H

    def sample(self, n, seed=3, return_latent=False):
        """
        Sample by blocked Gibbs sampling
        """
        B = self.B
        b = self.b
        c = self.c
        dh = len(c)
        dx = len(b)

        # Initialize the state of the Markov chain
        with util.NumpySeedContext(seed=seed):
            X = np.random.randn(n, dx)
            H = np.random.randint(1, 2, (n, dh))*2 - 1.0
        # burn-in
        for t in range(self.burnin):
            X, H = self._blocked_gibbs_next(X, H)
        # sampling
        X, H = self._blocked_gibbs_next(X, H)
        if return_latent:
            return Data(X), H
        else:
            return Data(X)


