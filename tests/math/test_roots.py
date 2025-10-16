# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
import pytest
from numpy import isclose

from polykin.math.roots import (fixpoint_anderson, fzero_brent, fzero_newton,
                                fzero_secant)


def f(x):
    "Test for scalar root-finding methods."
    return 2*x**3 + 4*x**2 + x - 2


f.sol = min(np.roots((2, 4, 1, -2)), key=lambda x: abs(x - 0.5))


def g(x):
    "Test for vector fixed-point solution methods."
    x1, x2 = x
    return np.array([0.5*np.cos(x1) + 0.1*x2 + 0.5, np.sin(x2) - 0.2*x1 + 1.2])


g.sol = np.array([0.97458605, 1.93830731])


def test_fzero_newton():
    # stop xtol
    xtol = 1e-9
    res = fzero_newton(f, 1.5, xtol=xtol, ftol=1e-100)
    assert res.success
    assert "xtol" in res.message
    assert isclose(res.x, f.sol, atol=xtol)
    # stop ftol
    ftol = 1e-10
    res = fzero_newton(f, 1.5, xtol=1e-100, ftol=ftol)
    assert res.success
    assert "ftol" in res.message
    assert abs(res.f) < ftol
    # stop maxiter
    maxiter = 3
    res = fzero_newton(f, 1.5, maxiter=maxiter)
    assert not res.success
    assert "iterations" in res.message
    assert res.niter == maxiter


def test_fzero_secant():
    # stop xtol
    xtol = 1e-9
    res = fzero_secant(f, 1.5, 1.4, xtol=xtol, ftol=1e-100)
    assert res.success
    assert "xtol" in res.message
    assert isclose(res.x, f.sol, atol=xtol)
    # stop ftol
    ftol = 1e-10
    res = fzero_secant(f, 1.5, 1.4, xtol=1e-100, ftol=ftol)
    assert res.success
    assert "ftol" in res.message
    assert abs(res.f) < ftol
    # stop maxiter
    maxiter = 3
    res = fzero_secant(f, 1.5, 1.4, maxiter=maxiter)
    assert not res.success
    assert "iterations" in res.message
    assert res.niter == maxiter


def test_fzero_brent():
    # stop xtol
    xtol = 1e-9
    res = fzero_brent(f, 0.1, 1.0, xtol=xtol, ftol=1e-100)
    assert res.success
    assert "xtol" in res.message
    assert isclose(res.x, f.sol, atol=xtol)
    # stop ftol
    ftol = 1e-10
    res = fzero_brent(f, 0.1, 1.0, xtol=1e-100, ftol=ftol)
    assert res.success
    assert "ftol" in res.message
    assert abs(res.f) < ftol
    # stop maxiter
    maxiter = 3
    res = fzero_brent(f, 0.1, 1.0, maxiter=maxiter)
    assert not res.success
    assert "iterations" in res.message
    assert res.niter == maxiter
    # no change of sign in interval
    with pytest.raises(ValueError):
        _ = fzero_brent(f, 0.1, 0.2)


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
