# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
import pytest
from numpy import isclose

from polykin.math.roots import fzero_brent, fzero_newton, fzero_secant


def f(x):
    "Test for scalar root-finding methods."
    return 2*x**3 + 4*x**2 + x - 2


f.sol = min(np.roots((2, 4, 1, -2)), key=lambda x: abs(x - 0.5))


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
