# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""
This module provides methods to create, visualize, fit, combine, integrate,
etc. theoretical and experimental chain-length distributions.
"""

from polykin.distributions.analyticaldistributions import *
from polykin.distributions.datadistribution import *
from polykin.distributions.baseclasses import *
from polykin.distributions.sampledata import sample_mmd

from polykin._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
