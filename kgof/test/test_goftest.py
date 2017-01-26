"""
Module for testing goftest module.
"""

__author__ = 'wittawat'

import numpy as np
import matplotlib.pyplot as plt
import kgof.data as data
import kgof.util as util
import kgof.kernel as kernel
import kgof.goftest as gof
import kgof.glo as glo
import scipy.stats as stats

import unittest


class TestFSSD(unittest.TestCase):
    def setUp(self):
        #n = 300
        #dx = 2
        #pdata_mean = get_pdata_mean(n, dx)
        #X, Y = pdata_mean.xy()
        #gwx2 = util.meddistance(X)**2
        #gwy2 = util.meddistance(Y)**2
        #J = 2
        #V = np.random.randn(J, dx)
        #W = np.random.randn(J, 1)

        #self.gnfsic = it.GaussNFSIC(gwx2, gwy2, V, W, alpha=0.01)
        #self.pdata_mean = pdata_mean
        pass

    def test_perform_test(self):
        #test_result = self.gnfsic.perform_test(self.pdata_mean)
        # should reject. Cannot assert this for sure.
        #self.assertTrue(test_result['h0_rejected'], 'Test should reject H0')
        pass 

    def test_compute_stat(self):
        #stat = self.gnfsic.compute_stat(self.pdata_mean)
        #self.assertGreater(stat, 0)
        pass


    def tearDown(self):
        pass


if __name__ == '__main__':
   unittest.main()

