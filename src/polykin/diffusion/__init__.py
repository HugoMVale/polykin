# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""
This module provides methods to calculate mutual and self-diffusion
coefficients in binary liquid and gas mixtures.

For illustration examples, please refer to the associated
[tutorial](/polykin/tutorials/diffusion_coefficients).
"""

from polykin.diffusion.vrentasduda import *
from polykin.diffusion.estimation_methods import *

from polykin._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
