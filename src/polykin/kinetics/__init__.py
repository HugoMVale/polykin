# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""
This module provides methods to create and visualize the types of kinetic
coefficients most often used in polymer reactor models.

For illustration examples, please refer to the associated
[tutorial](/polykin/tutorials/kinetic_coefficients/).
"""

from polykin.kinetics.thermal import *
from polykin.kinetics.cld import *

from polykin._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
