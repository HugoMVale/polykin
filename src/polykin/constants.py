# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

"""Physical constants used throughout PolyKin.

This module re-exports selected constants from :mod:`scipy.constants` with
short, conventional names.
"""

from scipy.constants import (
    Avogadro,
    Boltzmann,
    Planck,
    gas_constant,
    gravitational_constant,
)

__all__ = ["g", "h", "kB", "NA", "R"]

g = gravitational_constant
h = Planck
kB = Boltzmann
NA = Avogadro
R = gas_constant
