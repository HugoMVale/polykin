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
huge = float(_finfo.max)
tiny = float(_finfo.tiny)
sqrt_eps = math.sqrt(eps)
