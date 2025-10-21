# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import allclose

from polykin.math.fixpoint import fixpoint_anderson, fixpoint_wegstein


def g(x):
    "Test for vector fixed-point solution methods."
    x1, x2 = x
    return np.array([0.5*np.cos(x1) + 0.1*x2 + 0.5, np.sin(x2) - 0.2*x1 + 1.2])


g.sol = np.array([0.97458605, 1.93830731])


def test_fixpoint_anderson():
    # stop tolx
    tolx = 1e-13
    res = fixpoint_anderson(g, np.array([0.0, 0.0]), m=2, tolx=tolx)
    assert res.success
    assert "tolx" in res.message
    assert allclose(res.x, g.sol, atol=tolx)
    assert allclose(res.f, g(res.x) - res.x)
    # stop maxiter
    maxiter = 3
    res = fixpoint_anderson(g, np.array([0.0, 0.0]), maxiter=maxiter)
    assert not res.success
    assert "iterations" in res.message
    assert res.niter == maxiter
    assert allclose(res.f, g(res.x) - res.x)


def test_fixpoint_wegstein():
    # stop tolx, normal
    tolx = 1e-8
    res = fixpoint_wegstein(g, np.array([0.0, 0.0]), qmax=0.5, tolx=tolx)
    assert res.success
    assert "tolx" in res.message
    assert allclose(res.x, g.sol, atol=tolx)
    assert allclose(res.f, g(res.x) - res.x)
    # stop tolx, no acceleration
    tolx = 1e-8
    res = fixpoint_wegstein(g, np.array([0.0, 0.0]), wait=100, tolx=tolx)
    assert res.success
    assert "tolx" in res.message
    assert allclose(res.x, g.sol, atol=tolx)
    assert allclose(res.f, g(res.x) - res.x)
    # stop maxiter
    maxiter = 3
    res = fixpoint_wegstein(g, np.array([0.0, 0.0]), maxiter=maxiter)
    assert not res.success
    assert "iterations" in res.message
    assert res.niter == maxiter
    assert allclose(res.f, g(res.x) - res.x)
