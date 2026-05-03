# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

import math

import numpy as np

__all__ = [
    "eps",
    "huge",
    "tiny",
    "sqrt_eps",
]

_finfo = np.finfo(np.float64)

eps = float(_finfo.eps)
"""Machine epsilon for double-precision floating-point numbers."""

huge = float(_finfo.max)
"""Largest finite double-precision floating-point number."""

tiny = float(_finfo.tiny)
"""Smallest positive normalized double-precision floating-point number."""

sqrt_eps = math.sqrt(eps)
"""Square root of machine epsilon."""
