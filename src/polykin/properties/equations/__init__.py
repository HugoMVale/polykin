# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""
This modules implements commonly used equations (correlations) to describe the
physical properties of pure components.
"""

from .base import plotequations
from .dippr import *
from .vapor_pressure import *
from .viscosity import *
