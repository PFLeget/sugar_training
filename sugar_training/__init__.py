#!/usr/bin/env python

"""
Some description.
"""

#import os
#import glob

# Automatically import all modules (python files)
#__all__ = [os.path.basename(m).replace('.py', '') for m in glob.glob("sugar/*.py")
#           if '__init__' not in m]

from .mean_gp import comp_mean
from .gaussian_process import load_data_bin_gp
from .gaussian_process import gp_sed 

from .math_toolbox import passage
from .math_toolbox import passage_error
from .math_toolbox import svd_inverse
from .math_toolbox import cholesky_inverse
from .math_toolbox import biweight_M
from .math_toolbox import biweight_S

from .extinction import extinctionLaw
