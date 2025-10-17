# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
import pytest
from numpy import isclose

from polykin.math.fixpoint import fixpoint_anderson


def g(x):
    "Test for vector fixed-point solution methods."
    x1, x2 = x
    return np.array([0.5*np.cos(x1) + 0.1*x2 + 0.5, np.sin(x2) - 0.2*x1 + 1.2])


g.sol = np.array([0.97458605, 1.93830731])


def test_fixpoint_anderson():
    # stop xtol
    xtol = 1e-9
    res = fixpoint_anderson(g, np.array([0.0, 0.0]), xtol=xtol)
    assert res.success
    assert "xtol" in res.message
    assert np.allclose(res.x, g.sol, atol=xtol)
    # stop maxiter
    maxiter = 3
    res = fixpoint_anderson(g, np.array([0.0, 0.0]), maxiter=maxiter)
    assert not res.success
    assert "iterations" in res.message
    assert res.niter == maxiter
