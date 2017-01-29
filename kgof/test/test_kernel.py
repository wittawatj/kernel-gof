"""
Module for testing kernel module.
"""

__author__ = 'wittawat'

import numpy as np
import matplotlib.pyplot as plt
import kgof.data as data
import kgof.density as density
import kgof.util as util
import kgof.kernel as kernel
import kgof.goftest as gof
import kgof.glo as glo
import scipy.stats as stats
import tensorflow as tf

import unittest


class TestKGauss(unittest.TestCase):
    def setUp(self):
        pass

    def test_basic(self):
        """
        Nothing special. Just test basic things.
        """
        # sample
        n = 10
        d = 3
        with util.NumpySeedContext(seed=29):
            X = np.random.randn(n, d)*3
            k = kernel.KGauss(sigma2=1)
            K = k.eval(X, X)

            self.assertEqual(K.shape, (n, n))
            self.assertTrue(np.all(K >= 0-1e-6))
            self.assertTrue(np.all(K <= 1+1e-6), 'K not bounded by 1')

    def test_gradX_y(self):
        n = 10
        with util.NumpySeedContext(seed=10):
            for d in [1, 3]:
                y = np.random.randn(d)*2
                X = np.random.rand(n, d)*3

                sigma2 = 1.3
                k = kernel.KGauss(sigma2=sigma2)
                # n x d
                G = k.gradX_y(X, y)
                # check correctness 
                K = k.eval(X, y[np.newaxis, :])
                myG = -K/sigma2*(X-y)

                self.assertEqual(G.shape, myG.shape)
                np.testing.assert_almost_equal(G, myG)

    def test_gradXY_sum(self):
        n = 11
        with util.NumpySeedContext(seed=12):
            for d in [3, 1]:
                X = np.random.randn(n, d)
                sigma2 = 1.4
                k = kernel.KGauss(sigma2=sigma2)

                # n x n
                myG = np.zeros((n, n))
                K = k.eval(X, X)
                for i in range(n):
                    for j in range(n):
                        diffi2 = np.sum( (X[i, :] - X[j, :])**2 )
                        #myG[i, j] = -diffi2*K[i, j]/(sigma2**2)+ d*K[i, j]/sigma2
                        myG[i, j] = K[i, j]/sigma2*(d - diffi2/sigma2)

                # check correctness 
                G = k.gradXY_sum(X, X)

                self.assertEqual(G.shape, myG.shape)
                np.testing.assert_almost_equal(G, myG)

    #def test_gradX_Y_avgX(self):
    #    nx = 11
    #    ny = 5
    #    with util.NumpySeedContext(seed=14):
    #        for d in [3, 1]:
    #            X = np.random.randn(nx, d) + 1
    #            Y = np.random.rand(ny, d)*2

    #            sigma2 = 1.7
    #            k = kernel.KGauss(sigma2=sigma2)

    #            # nx x ny x d
    #            T = np.zeros((nx, ny, d))
    #            K = k.eval(X, Y)
    #            for i in range(nx):
    #                for j in range(ny):
    #                    Di = X[i, :] - Y[j, :]
    #                    T[i, j, :] = -K[i, j]/sigma2*Di
    #            myG = np.mean(T, axis=0)

    #            # check correctness 
    #            G = k.gradX_Y_avgX(X, Y)

    #            self.assertEqual(G.shape, myG.shape)
    #            np.testing.assert_almost_equal(G, myG)

    def tearDown(self):
        pass


if __name__ == '__main__':
   unittest.main()

