"""Simulation to examine the P(reject) as the parameters for each problem are 
varied. What varies will depend on the problem."""

__author__ = 'wittawat'

import kgof
import kgof.data as data
import kgof.glo as glo
import kgof.density as density
import kgof.goftest as gof
import kgof.util as util 
import kgof.kernel as kernel 

# need independent_jobs package 
# https://github.com/karlnapf/independent-jobs
# The independent_jobs and kgof have to be in the global search path (.bashrc)
import independent_jobs as inj
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.Log import logger
import logging
import math
#import numpy as np
import autograd.numpy as np
import os
import sys 
import time

"""
All the job functions return a dictionary with the following keys:
    - goftest: independence test object. (may or may not return)
    - test_result: the result from calling perform_test(te).
    - time_secs: run time in seconds 
"""

def job_fssdJ1_med(p, data_source, tr, te, r, J=1):
    """
    FSSD test with a Gaussian kernel, where the test locations are randomized,
    and the Gaussian width is set with the median heuristic. Use full sample.
    No training/testing splits.

    p: an UnnormalizedDensity
    data_source: a DataSource
    tr, te: Data
    r: trial number (positive integer)
    """

    # full data
    data = tr + te
    X = data.data()
    with util.ContextTimer() as t:
        # median heuristic 
        med = util.meddistance(X, subsample=1000)
        k = kernel.KGauss(med**2)
        V = util.fit_gaussian_draw(X, J, seed=r+1)

        fssd_med = gof.FSSD(p, k, V, alpha=alpha, n_simulate=2000, seed=r)
        fssd_med_result = fssd_med.perform_test(data)
    return { 'test_result': fssd_med_result, 'time_secs': t.secs}

def job_fssdJ2_med(p, data_source, tr, te, r):
    """
    FSSD. J=2
    """
    return job_fssdJ1_med(p, data_source, tr, te, r, J=2)

def job_fssdJ5_med(p, data_source, tr, te, r):
    """
    FSSD. J=2
    """
    return job_fssdJ1_med(p, data_source, tr, te, r, J=5)

def job_fssdJ1_opt(p, data_source, tr, te, r, J=1):
    """
    FSSD with optimization on tr. Test on te. Use a Gaussian kernel.
    """
    Xtr = tr.data()
    with util.ContextTimer() as t:
        # Use grid search to initialize the gwidth
        n_gwidth_cand = 5
        gwidth_factors = 2.0**np.linspace(-3, 3, n_gwidth_cand) 
        med2 = util.meddistance(Xtr, 1000)**2

        k = kernel.KGauss(med2*2)
        # fit a Gaussian to the data and draw to initialize V0
        V0 = util.fit_gaussian_draw(Xtr, J, seed=r+1, reg=1e-6)
        list_gwidth = np.hstack( ( (med2)*gwidth_factors ) )
        besti, objs = gof.GaussFSSD.grid_search_gwidth(p, tr, V0, list_gwidth)
        gwidth = list_gwidth[besti]
        assert util.is_real_num(gwidth), 'gwidth not real. Was %s'%str(gwidth)
        assert gwidth > 0, 'gwidth not positive. Was %.3g'%gwidth
        logging.info('After grid search, gwidth=%.3g'%gwidth)
        
        ops = {
            'reg': 1e-2,
            'max_iter': 50,
            'tol_fun': 1e-3,
            'disp': True,
            'locs_bounds_frac':1.0,
            'gwidth_lb': 1e-2,
            }

        V_opt, gwidth_opt, info = gof.GaussFSSD.optimize_locs_widths(p, tr,
                gwidth, V0, **ops) 
        # Use the optimized parameters to construct a test
        k_opt = kernel.KGauss(gwidth_opt)
        fssd_opt = gof.FSSD(p, k_opt, V_opt, alpha=alpha, n_simulate=2000, seed=r)
        fssd_opt_result = fssd_opt.perform_test(te)
    return {'test_result': fssd_opt_result, 'time_secs': t.secs, 
            'goftest': fssd_opt, 'opt_info': info,
            }

def job_fssdJ5_opt(p, data_source, tr, te, r):
    return job_fssdJ1_opt(p, data_source, tr, te, r, J=5)

def job_kstein_med(p, data_source, tr, te, r):
    # full data
    data = tr + te
    X = data.data()
    with util.ContextTimer() as t:
        # median heuristic 
        med = util.meddistance(X, subsample=1000)
        k = kernel.KGauss(med**2)

        kstein = gof.KernelSteinTest(p, k, alpha=alpha, n_simulate=500, seed=r)
        kstein_result = kstein.perform_test(data)
    return { 'test_result': kstein_result, 'time_secs': t.secs}


# Define our custom Job, which inherits from base class IndependentJob
class Ex2Job(IndependentJob):
   
    def __init__(self, aggregator, p, data_source,
            prob_label, rep, job_func, prob_param):
        #walltime = 60*59*24 
        walltime = 60*59
        memory = int(tr_proportion*sample_size*1e-2) + 50

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                               memory=memory)
        # p: an UnnormalizedDensity
        self.p = p
        self.data_source = data_source
        self.prob_label = prob_label
        self.rep = rep
        self.job_func = job_func
        self.prob_param = prob_param

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):

        p = self.p
        data_source = self.data_source 
        r = self.rep
        prob_param = self.prob_param
        job_func = self.job_func
        # sample_size is a global variable
        data = data_source.sample(sample_size, seed=r)
        with util.ContextTimer() as t:
            tr, te = data.split_tr_te(tr_proportion=tr_proportion, seed=r+21 )
            prob_label = self.prob_label
            logger.info("computing. %s. prob=%s, r=%d,\
                    param=%.3g"%(job_func.__name__, prob_label, r, prob_param))


            job_result = job_func(p, data_source, tr, te, r)

            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = job_func.__name__
        logger.info("done. ex2: %s, prob=%s, r=%d, param=%.3g. Took: %.3g s "%(func_name,
            prob_label, r, prob_param, t.secs))

        # save result
        fname = '%s-%s-n%d_r%d_p%.3f_a%.3f_trp%.2f.p' \
                %(prob_label, func_name, sample_size, r, prob_param, alpha,
                        tr_proportion)
        glo.ex_save_result(ex, job_result, prob_label, fname)


# This import is needed so that pickle knows about the class Ex2Job.
# pickle is used when collecting the results from the submitted jobs.
from kgof.ex.ex2_prob_params import Ex2Job
from kgof.ex.ex2_prob_params import job_fssdJ1_med
from kgof.ex.ex2_prob_params import job_fssdJ2_med
from kgof.ex.ex2_prob_params import job_fssdJ5_med
from kgof.ex.ex2_prob_params import job_fssdJ1_opt
from kgof.ex.ex2_prob_params import job_fssdJ5_opt
from kgof.ex.ex2_prob_params import job_kstein_med

#--- experimental setting -----
ex = 2

# sample size = n (the training and test sizes are n/2)
sample_size = 500

# number of test locations / test frequencies J
alpha = 0.05
tr_proportion = 0.5
# repetitions for each parameter setting
reps = 100

method_job_funcs = [ 
        job_fssdJ1_med, job_fssdJ5_med, 
        job_fssdJ1_opt, 
        job_fssdJ5_opt,
        job_kstein_med, 
       ]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting of (pi, r) already exists.
is_rerun = False
#---------------------------

def get_pqsource_list(prob_label):
    """
    Return [(prob_param, p, ds) for ... ], a list of tuples
    where 
    - prob_param: a problem parameters. Each parameter has to be a
      scalar (so that we can plot them later). Parameters are preferably
      positive integers.
    - p: a Density representing the distribution p
    - ds: a DataSource, each corresponding to one parameter setting.
    """
    sg_ds = [1, 5, 10, 15]
    gmd_ds = [5, 20, 40, 60]
    gvinc_d1_vs = [1, 2, 3, 4] 
    gvinc_d5_vs = [1, 2, 3, 4]
    prob2tuples = { 
            # H0 is true. vary d. P = Q = N(0, I)
            'sg': [(d, density.IsotropicNormal(np.zeros(d), 1),
                data.DSIsotropicNormal(np.zeros(d), 1) ) for d in sg_ds],

            # vary d. P = N(0, I), Q = N( (1,..0), I)
            'gmd': [(d, density.IsotropicNormal(np.zeros(d), 1),
                data.DSIsotropicNormal(np.hstack((1, np.zeros(d-1))), 1) ) 
                for d in gmd_ds
                ],
            # d=1. Increase the variance. P = N(0, I). Q = N(0, v*I)
            'gvinc_d1': [(var, density.IsotropicNormal(np.zeros(1), 1),
                data.DSIsotropicNormal(np.zeros(1), var) ) 
                for var in gvinc_d1_vs
                ],
            # d=5. Increase the variance. P = N(0, I). Q = N(0, v*I)
            'gvinc_d5': [(var, density.IsotropicNormal(np.zeros(5), 1),
                data.DSIsotropicNormal(np.zeros(5), var) ) 
                for var in gvinc_d5_vs
                ],
            }
    if prob_label not in prob2tuples:
        raise ValueError('Unknown problem label. Need to be one of %s'%str(prob2tuples.keys()) )
    return prob2tuples[prob_label]


def run_problem(prob_label):
    """Run the experiment"""
    L = get_pqsource_list(prob_label)
    prob_params, ps, data_sources = zip(*L)
    # make them lists 
    prob_params = list(prob_params)
    ps = list(ps)
    data_sources = list(data_sources)

    # ///////  submit jobs //////////
    # create folder name string
    #result_folder = glo.result_folder()
    from kgof.config import expr_configs
    tmp_dir = expr_configs['scratch_path']
    foldername = os.path.join(tmp_dir, 'kgof_slurm', 'e%d'%ex)
    logger.info("Setting engine folder to %s" % foldername)

    # create parameter instance that is needed for any batch computation engine
    logger.info("Creating batch parameter instance")
    batch_parameters = BatchClusterParameters(
        foldername=foldername, job_name_base="e%d_"%ex, parameter_prefix="")

    # Use the following line if Slurm queue is not used.
    #engine = SerialComputationEngine()
    engine = SlurmComputationEngine(batch_parameters)
    n_methods = len(method_job_funcs)
    # repetitions x len(prob_params) x #methods
    aggregators = np.empty((reps, len(prob_params), n_methods ), dtype=object)
    for r in range(reps):
        for pi, param in enumerate(prob_params):
            for mi, f in enumerate(method_job_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-n%d_r%d_p%.3f_a%.3f_trp%.2f.p' \
                    %(prob_label, func_name, sample_size, r, param, alpha,
                            tr_proportion)
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    job_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(job_result))
                    aggregators[r, pi, mi] = sra
                else:
                    # result not exists or rerun

                    # p: an UnnormalizedDensity object
                    p = ps[pi]
                    job = Ex2Job(SingleResultAggregator(), p, data_sources[pi],
                            prob_label, r, f, param)
                    agg = engine.submit_job(job)
                    aggregators[r, pi, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    job_results = np.empty((reps, len(prob_params), n_methods), dtype=object)
    for r in range(reps):
        for pi, param in enumerate(prob_params):
            for mi, f in enumerate(method_job_funcs):
                logger.info("Collecting result (%s, r=%d, param=%.3g)" %
                        (f.__name__, r, param))
                # let the aggregator finalize things
                aggregators[r, pi, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                job_result = aggregators[r, pi, mi].get_final_result().result
                job_results[r, pi, mi] = job_result

    #func_names = [f.__name__ for f in method_job_funcs]
    #func2labels = exglobal.get_func2label_map()
    #method_labels = [func2labels[f] for f in func_names if f in func2labels]

    # save results 
    results = {'job_results': job_results, 'prob_params': prob_params, 
            'alpha': alpha, 'repeats': reps, 
            'ps': ps,
            'list_data_source': data_sources, 
            'tr_proportion': tr_proportion,
            'method_job_funcs': method_job_funcs, 'prob_label': prob_label,
            'sample_size': sample_size, 
            }
    
    # class name 
    fname = 'ex%d-%s-me%d_n%d_rs%d_pmi%.3f_pma%.3f_a%.3f_trp%.2f.p' \
        %(ex, prob_label, n_methods, sample_size, reps, min(prob_params),
                max(prob_params), alpha, tr_proportion)

    glo.ex_save_result(ex, results, fname)
    logger.info('Saved aggregated results to %s'%fname)


def main():
    if len(sys.argv) != 2:
        print('Usage: %s problem_label'%sys.argv[0])
        sys.exit(1)
    prob_label = sys.argv[1]

    run_problem(prob_label)

if __name__ == '__main__':
    main()

