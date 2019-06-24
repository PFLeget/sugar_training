#!/usr/bin/env python

"""
Some description.
"""

from .load_and_write import load_pickle
from .load_and_write import write_pickle

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
from .math_toolbox import loess
from .math_toolbox import correlation_weighted
from .math_toolbox import correlation_significance
from .math_toolbox import neff_weighted
from .math_toolbox import flbda2ABmag

from .cosmology import luminosity_distance

from .extinction import extinctionLaw
from .emfa_analysis import run_emfa_analysis

from .sed_fitting import multilinearfit
from .sed_fitting import sugar_fitting
from .make_sugar_sed import make_sugar

from .write_sugar_template import write_sugar

from .compute_sugar_parameter import comp_sugar_param

from . import plotting
