# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""
This module provides means to evaluate physical property equations and
estimate physical properties often used in polymer reactor models.

For illustration examples, please refer to the associated
[tutorial](/polykin/tutorials/physprop_equations).
"""

from polykin.physprops.dippr import *
from polykin.physprops.tait import *


from polykin._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
