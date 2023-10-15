# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""
This module provides methods to create, visualize, fit, combine, integrate,
etc. theoretical and experimental chain-length distributions.

For illustration examples, please refer to the associated
[tutorial](/polykin/tutorials/distributions).
"""

from .analyticaldistributions import *
from .datadistribution import *
from .base import *
from .sampledata import sample_mmd
