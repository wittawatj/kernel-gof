"""
Module containing data structures for representing datasets.
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import math
import numpy as np
import kgof.util as util
#import scipy.stats as stats

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

    @abstractmethod
    def dim(self):
        """Return the dimension of the data. """
        raise NotImplementedError()
    

