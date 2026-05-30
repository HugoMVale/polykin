# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import allclose, isclose

from polykin.math.fixpoint import (
    fixpoint_anderson,
    fixpoint_damped,
    fixpoint_dem,
    fixpoint_steffensen,
    fixpoint_wegstein,
)


def g_vector(x):
    """Test for vector fixed-point solution methods."""
    x1, x2 = x
    return np.array([0.5 * np.cos(x1) + 0.1 * x2 + 0.5, np.sin(x2) - 0.2 * x1 + 1.2])


G_VECTOR_XS = np.array([0.97458605, 1.93830731])


def g_scalar(x):
    """Test for scalar fixed-point solution methods."""
    return np.cos(x)


G_SCALAR_XS = 0.7390851332151607

SUCCESS_ON_CALLBACK = False
NITER_ON_CALLBACK = 2


def callback(niter, x, fx):
    print(f"Iteration {niter}: x={x}, f(x)={fx}")
    return (niter >= NITER_ON_CALLBACK, SUCCESS_ON_CALLBACK)


def test_fixpoint_anderson():
    # stop tolx
    tolx = 1e-13
    for m in range(1, 5):
        sol = fixpoint_anderson(g_vector, np.array([0.0, 0.0]), m=m, tolx=tolx)
        assert sol.success
        assert "tolx" in sol.message
        assert allclose(sol.x, G_VECTOR_XS, atol=tolx)
        assert allclose(sol.f, g_vector(sol.x) - sol.x)
    # stop maxiter
    maxiter = 3
    sol = fixpoint_anderson(g_vector, np.array([0.0, 0.0]), maxiter=maxiter)
    assert not sol.success
    assert "iterations" in sol.message
    assert sol.niter == maxiter
    assert allclose(sol.f, g_vector(sol.x) - sol.x)
    # with callback
    sol = fixpoint_anderson(g_vector, np.array([0.0, 0.0]), m=2, callback=callback)
    assert "callback" in sol.message
    assert sol.success == SUCCESS_ON_CALLBACK
    assert sol.niter == NITER_ON_CALLBACK


def test_fixpoint_wegstein():
    # stop tolx, normal
    TOLX = 1e-8
    sol = fixpoint_wegstein(g_vector, np.array([0.0, 0.0]), qmax=0.5, tolx=TOLX)
    assert sol.success
    assert "tolx" in sol.message
    assert allclose(sol.x, G_VECTOR_XS, atol=TOLX)
    assert allclose(sol.f, g_vector(sol.x) - sol.x)
    # stop tolx, no acceleration
    sol = fixpoint_wegstein(g_vector, np.array([0.0, 0.0]), wait=100, tolx=TOLX)
    assert sol.success
    assert "tolx" in sol.message
    assert allclose(sol.x, G_VECTOR_XS, atol=TOLX)
    assert allclose(sol.f, g_vector(sol.x) - sol.x)
    # stop maxiter
    maxiter = 3
    sol = fixpoint_wegstein(g_vector, np.array([0.0, 0.0]), maxiter=maxiter)
    assert not sol.success
    assert "iterations" in sol.message
    assert sol.niter == maxiter
    assert allclose(sol.f, g_vector(sol.x) - sol.x)
    # consistency with fixpoint_damped
    q = 0
    sol_wegstein = fixpoint_wegstein(
        g_vector, np.array([0.0, 0.0]), qmin=q, qmax=q, tolx=TOLX
    )
    sol_damped = fixpoint_damped(g_vector, np.array([0.0, 0.0]), q=q, tolx=TOLX)
    assert allclose(sol_wegstein.x, sol_damped.x, atol=TOLX)
    assert allclose(sol_wegstein.f, sol_damped.f, atol=TOLX)
    assert sol_wegstein.niter == sol_damped.niter
    # with callback
    sol = fixpoint_wegstein(g_vector, np.array([0.0, 0.0]), callback=callback)
    assert "callback" in sol.message
    assert sol.success == SUCCESS_ON_CALLBACK
    assert sol.niter == NITER_ON_CALLBACK


def test_fixpoint_damped():
    # stop tolx
    tolx = 1e-8
    sol = fixpoint_damped(g_vector, np.array([0.0, 0.0]), q=0.2, tolx=tolx)
    assert sol.success
    assert "tolx" in sol.message
    assert allclose(sol.x, G_VECTOR_XS, atol=tolx)
    assert allclose(sol.f, g_vector(sol.x) - sol.x)
    # stop maxiter
    maxiter = 3
    sol = fixpoint_damped(g_vector, np.array([0.0, 0.0]), maxiter=maxiter)
    assert not sol.success
    assert "iterations" in sol.message
    assert sol.niter == maxiter
    assert allclose(sol.f, g_vector(sol.x) - sol.x)
    # with callback
    sol = fixpoint_damped(g_vector, np.array([0.0, 0.0]), callback=callback)
    assert "callback" in sol.message
    assert sol.success == SUCCESS_ON_CALLBACK
    assert sol.niter == NITER_ON_CALLBACK


def test_fixpoint_dem():
    # stop tolx
    tolx = 1e-8
    sol = fixpoint_dem(g_vector, np.array([0.0, 0.0]), q=0.2, tolx=tolx)
    assert sol.success
    assert "tolx" in sol.message
    assert allclose(sol.x, G_VECTOR_XS, atol=tolx)
    assert allclose(sol.f, g_vector(sol.x) - sol.x)
    # stop maxiter
    maxiter = 3
    sol = fixpoint_dem(g_vector, np.array([0.0, 0.0]), maxiter=maxiter)
    assert not sol.success
    assert "iterations" in sol.message
    assert sol.niter == maxiter
    assert allclose(sol.f, g_vector(sol.x) - sol.x)
    # with callback
    sol = fixpoint_dem(g_vector, np.array([0.0, 0.0]), callback=callback)
    assert "callback" in sol.message
    assert sol.success == SUCCESS_ON_CALLBACK
    assert sol.niter == NITER_ON_CALLBACK


def test_fixpoint_steffensen():
    tolx = 1e-10
    sol = fixpoint_steffensen(g_scalar, 0.5, tolx=tolx)
    assert sol.success
    assert "tolx" in sol.message
    assert isclose(sol.x, G_SCALAR_XS, atol=tolx)
    assert isclose(sol.f, g_scalar(sol.x) - sol.x, atol=tolx)
    # stop maxiter
    maxiter = 1
    sol = fixpoint_steffensen(g_scalar, 0.5, maxiter=maxiter)
    assert not sol.success
    assert "iterations" in sol.message
    assert sol.niter == maxiter
    assert isclose(sol.f, g_scalar(sol.x) - sol.x, atol=tolx)
    # test denominator zero
    sol = fixpoint_steffensen(lambda x: x + 1.0, 0.0)
    assert not sol.success
    assert "denominator" in sol.message
    assert sol.niter == 1
    assert allclose(sol.f, 1.0)
    # with callback
    sol = fixpoint_steffensen(g_scalar, 0.5, tolx=tolx, callback=callback)
    assert "callback" in sol.message
    assert sol.success == SUCCESS_ON_CALLBACK
    assert sol.niter == NITER_ON_CALLBACK
