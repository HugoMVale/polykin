# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import allclose

from polykin.math.fixpoint import fixpoint_anderson, fixpoint_wegstein


def g(x):
    """Test for vector fixed-point solution methods."""
    x1, x2 = x
    return np.array([0.5 * np.cos(x1) + 0.1 * x2 + 0.5, np.sin(x2) - 0.2 * x1 + 1.2])


g.xs = np.array([0.97458605, 1.93830731])


def test_fixpoint_anderson():
    # stop tolx
    tolx = 1e-13
    for m in range(1, 5):
        sol = fixpoint_anderson(g, np.array([0.0, 0.0]), m=m, tolx=tolx)
        assert sol.success
        assert "tolx" in sol.message
        assert allclose(sol.x, g.xs, atol=tolx)
        assert allclose(sol.f, g(sol.x) - sol.x)
    # stop maxiter
    maxiter = 3
    sol = fixpoint_anderson(g, np.array([0.0, 0.0]), maxiter=maxiter)
    assert not sol.success
    assert "iterations" in sol.message
    assert sol.niter == maxiter
    assert allclose(sol.f, g(sol.x) - sol.x)
    # initial guess close to solution
    sol = fixpoint_anderson(g, g.xs)
    assert sol.success
    assert sol.niter == 0


def test_fixpoint_wegstein():
    # stop tolx, normal
    tolx = 1e-8
    sol = fixpoint_wegstein(g, np.array([0.0, 0.0]), qmax=0.5, tolx=tolx)
    assert sol.success
    assert "tolx" in sol.message
    assert allclose(sol.x, g.xs, atol=tolx)
    assert allclose(sol.f, g(sol.x) - sol.x)
    # stop tolx, no acceleration
    tolx = 1e-8
    sol = fixpoint_wegstein(g, np.array([0.0, 0.0]), wait=100, tolx=tolx)
    assert sol.success
    assert "tolx" in sol.message
    assert allclose(sol.x, g.xs, atol=tolx)
    assert allclose(sol.f, g(sol.x) - sol.x)
    # stop maxiter
    maxiter = 3
    sol = fixpoint_wegstein(g, np.array([0.0, 0.0]), maxiter=maxiter)
    assert not sol.success
    assert "iterations" in sol.message
    assert sol.niter == maxiter
    assert allclose(sol.f, g(sol.x) - sol.x)
