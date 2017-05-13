"""
Module containing the MMD two-sample test of Gretton et al., 2012 
"A Kernel Two-Sample Test" disguised as goodness-of-fit tests. Require the
ability to sample from the specified density. This module depends on an external
package

freqopttest https://github.com/wittawatj/interpretable-test

providing an implementation to the MMD test.

"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import autograd
import autograd.numpy as np
# Require freqopttest https://github.com/wittawatj/interpretable-test
import freqopttest.tst as tst
import freqopttest.data as fdata
import kgof.data as data
import kgof.goftest as gof
import kgof.util as util
import kgof.kernel as kernel
import logging
import matplotlib.pyplot as plt

import scipy
import scipy.stats as stats

class QuadMMDGof(gof.GofTest):
    """
    Goodness-of-fit test by drawing sample from the density p and test with
    the MMD test of Gretton et al., 2012. 

    H0: the sample follows p
    H1: the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, k, n_permute=400, alpha=0.01, seed=28):
        """
        p: an instance of UnnormalizedDensity
        k: an instance of Kernel
        n_permute: number of times to permute the samples to simulate from the 
            null distribution (permutation test)
        alpha: significance level 
        seed: random seed
        """
        super(QuadMMDGof, self).__init__(p, alpha)
        # Construct the MMD test
        self.mmdtest = tst.QuadMMDTest(k, n_permute=n_permute, alpha=alpha)
        self.k = k
        self.seed = seed
        ds = p.get_datasource()
        if ds is None:
            raise ValueError('%s test requires a density p which implements get_datasource(', str(QuadMMDGof))


    def perform_test(self, dat):
        """
        dat: an instance of Data
        """
        with util.ContextTimer() as t:
            seed = self.seed
            mmdtest = self.mmdtest
            p = self.p

            # Draw sample from p. #sample to draw is the same as that of dat
            ds = p.get_datasource()
            p_sample = ds.sample(dat.sample_size(), seed=seed)

            # Run the two-sample test on p_sample and dat
            # Make a two-sample test data
            tst_data = fdata.TSTData(p_sample.data(), dat.data())
            # Test 
            results = mmdtest.perform_test(tst_data)

        results['time_secs'] = t.secs
        return results

    def compute_stat(self, dat):
        mmdtest = self.mmdtest
        p = self.p
        # Draw sample from p. #sample to draw is the same as that of dat
        ds = p.get_datasource()
        p_sample = ds.sample(dat.sample_size(), seed=self.seed)

        # Make a two-sample test data
        tst_data = fdata.TSTData(p_sample.data(), dat.data())
        s = mmdtest.compute_stat(tst_data)
        return s

        
# end QuadMMDGof
