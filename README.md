# kgof

[![Build Status](https://travis-ci.org/wittawatj/kernel-gof.svg?branch=master)](https://travis-ci.org/wittawatj/kernel-gof)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/wittawatj/fsic-test/blob/master/LICENSE)

This repository contains a Python 2.7 implementation of the nonparametric
linear-time goodness-of-fit test described in  [our paper](https://arxiv.org/abs/1705.07673)

    A Linear-Time Kernel Goodness-of-Fit Test
    Wittawat Jitkrittum, Wenkai Xu, Zoltan Szabo, Kenji Fukumizu, Arthur Gretton
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
