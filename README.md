# kgof

[![Build Status](https://travis-ci.org/wittawatj/kernel-gof.svg?branch=master)](https://travis-ci.org/wittawatj/kernel-gof)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/wittawatj/kernel-gof/blob/master/LICENSE)

**11th July 2019**: For an implementation of our test in Julia, see [this
repository](https://github.com/torfjelde/KernelGoodnessOfFit.jl) by [Tor Erlend Fjelde](http://retiredparkingguard.com/).

**UPDATE**: On 8th Mar 2018, we have updated the code to support Python 3 (with
`futurize`). If you find any problem, please let us know. Thanks.

This repository contains a Python 2.7/3 implementation of the nonparametric
linear-time goodness-of-fit test described in  [our paper](https://arxiv.org/abs/1705.07673)

    A Linear-Time Kernel Goodness-of-Fit Test
    Wittawat Jitkrittum, Wenkai Xu, Zoltan Szabo, Kenji Fukumizu, Arthur Gretton
    NIPS 2017 (Best paper)
    https://arxiv.org/abs/1705.07673

## How to install?

The package can be installed with the `pip` command.

    pip install git+https://github.com/wittawatj/kernel-gof.git

Once installed, you should be able to do `import kgof` without any error.
`pip` will also resolve the following dependency automatically.

## Dependency

The following Python packages were used during development. Ideally, the
following packages with the specified version numbers or newer should be used.
However, older versions may work as well. We did not specifically rely on
newest features in these specified versions.

    autograd == 1.1.7
    matplotlib == 2.0.0
    numpy == 1.11.3
    scipy == 0.19.0

## Demo

To get started, check
[demo_kgof.ipynb](https://github.com/wittawatj/kernel-gof/blob/master/ipynb/demo_kgof.ipynb).
This is a Jupyter notebook which will guide you through from the beginning. It
can also be viewed on the web. There are many Jupyter notebooks in `ipynb`
folder demonstrating other implemented tests. Be sure to check them if you
would like to explore.

## Reproduce experimental results

Each experiment is defined in its own Python file with a name starting with
`exXX` where `XX` is a number. All the experiment files are in `kgof/ex`
folder. Each file is runnable with a command line argument. For example in
`ex1_vary_n.py`, we aim to check the test power of each testing algorithm
as a function of the sample size `n`. The script `ex1_vary_n.py` takes a
dataset name as its argument. See `run_ex1.sh` which is a standalone Bash
script on how to execute  `ex1_power_vs_n.py`.

We used [independent-jobs](https://github.com/karlnapf/independent-jobs)
package to parallelize our experiments over a
[Slurm](http://slurm.schedmd.com/) cluster (the package is not needed if you
just need to use our developed tests). For example, for
`ex1_vary_n.py`, a job is created for each combination of 

    (dataset, test algorithm, n, trial)

If you do not use Slurm, you can change the line 

    engine = SlurmComputationEngine(batch_parameters)

to 

    engine = SerialComputationEngine()

which will instruct the computation engine to just use a normal for-loop on a
single machine (will take a lot of time). Other computation engines that you
use might be supported. See  [independent-jobs's repository
page](https://github.com/karlnapf/independent-jobs).  Running simulation will
create a lot of result files (one for each tuple above) saved as Pickle. Also,
the `independent-jobs` package requires a scratch folder to save temporary
files for communication among computing nodes. Path to the folder containing
the saved results can be specified in `kgof/config.py` by changing the value of
`expr_results_path`:

    # Full path to the directory to store experimental results.
    'expr_results_path': '/full/path/to/where/you/want/to/save/results/',

The scratch folder needed by the `independent-jobs` package can be specified in
the same file by changing the value of `scratch_path`

    # Full path to the directory to store temporary files when running experiments
    'scratch_path': '/full/path/to/a/temporary/folder/',

To plot the results, see the experiment's corresponding Jupyter notebook in the
`ipynb/` folder. For example, for `ex1_vary_n.py` see `ipynb/ex1_results.ipynb`
to plot the results.


## Some note

* When adding a new `Kernel` or new `UnnormalizedDensity`, use `np.dot(X, Y)`
  instead of `X.dot(Y)`. `autograd` cannot differentiate the latter. Also, do
  not use `x += ...`.  Use `x = x + ..` instead.

* The sub-module `kgof.intertst` depends on the linear-time two-sample test of
  [Jitkrittum et al., 2016 (NIPS
  2016)](http://papers.nips.cc/paper/6148-interpretable-distribution-features-with-maximum-testing-power)
  implemented in  the `freqopttest` Python package which can be found
  [here](https://github.com/wittawatj/interpretable-test).


---------------

If you have questions or comments about anything related to this work, please
do not hesitate to contact [Wittawat Jitkrittum](http://wittawat.com).
