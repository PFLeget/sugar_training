#!/usr/bin/env python

"""Setup script."""

import glob
from setuptools import setup, find_packages

# Package name
name = 'sugar_training'

# Packages (subdirectories in clusters/)
packages = find_packages()

# Scripts (in scripts/)
scripts = glob.glob("scripts/*.py")

package_data = {}

setup(name=name,
      description=("sugar_training"),
      classifiers=["Topic :: Scientific :: Astronomy",
                   "Intended Audience :: Science/Research"],
      author="PFLeget",
      packages=packages,
      scripts=scripts)
