#!/usr/bin/env python

"""
Some description.
"""

#import os
#import glob

# Automatically import all modules (python files)
#__all__ = [os.path.basename(m).replace('.py', '') for m in glob.glob("sugar/*.py")
#           if '__init__' not in m]

from .load_data import load_data_sugar

from . import sugargp
from .mean_gp import comp_mean
from .gaussian_process import load_data_bin_gp
from .gaussian_process import gp_sed

from .math_toolbox import passage
from .math_toolbox import passage_error
from .math_toolbox import svd_inverse
from .math_toolbox import cholesky_inverse
from .math_toolbox import biweight_M
from .math_toolbox import biweight_S
from .math_toolbox import comp_rms

from .extinction import extinctionLaw
from .emfa_analysis import run_emfa_analysis

from .sed_fitting import multilinearfit
from .sed_fitting import sugar_fitting
from .make_sugar_sed import make_sugar

from . import plotting
