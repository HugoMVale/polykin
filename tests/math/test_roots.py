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
    # stop tolx
    tolx = 1e-9
    res = fzero_newton(f, 1.5, tolx=tolx, tolf=0.0)
    assert res.success
    assert "tolx" in res.message
    assert isclose(res.x, f.sol, atol=tolx)
    assert isclose(res.f, f(res.x))
    assert res.nfeval == res.niter
    # stop tolf
    tolf = 1e-10
    res = fzero_newton(f, 1.5, tolx=0.0, tolf=tolf)
    assert res.success
    assert "tolf" in res.message
    assert abs(res.f) <= tolf
    assert isclose(res.f, f(res.x))
    # stop maxiter
    maxiter = 3
    res = fzero_newton(f, 1.5, maxiter=maxiter)
    assert not res.success
    assert "iterations" in res.message
    assert res.niter == maxiter
    assert isclose(res.f, f(res.x))


def test_fzero_secant():
    # stop tolx
    tolx = 1e-9
    res = fzero_secant(f, 1.5, 1.4, tolx=tolx, tolf=0.0)
    assert res.success
    assert "tolx" in res.message
    assert isclose(res.x, f.sol, atol=tolx)
    assert isclose(res.f, f(res.x))
    assert res.nfeval == res.niter + 2
    # stop tolf
    tolf = 1e-10
    res = fzero_secant(f, 1.5, 1.4, tolx=0.0, tolf=tolf)
    assert res.success
    assert "tolf" in res.message
    assert abs(res.f) <= tolf
    assert isclose(res.f, f(res.x))
    # stop maxiter
    maxiter = 3
    res = fzero_secant(f, 1.5, 1.4, maxiter=maxiter)
    assert not res.success
    assert "iterations" in res.message
    assert res.niter == maxiter
    assert isclose(res.f, f(res.x))


def test_fzero_brent():
    # stop tolx
    tolx = 1e-9
    res = fzero_brent(f, 0.1, 1.0, tolx=tolx, tolf=0.0)
    assert res.success
    assert "tolx" in res.message
    assert isclose(res.x, f.sol, atol=tolx)
    assert isclose(res.f, f(res.x))
    # stop tolf
    tolf = 1e-10
    res = fzero_brent(f, 0.1, 1.0, tolx=0.0, tolf=tolf)
    assert res.success
    assert "tolf" in res.message
    assert abs(res.f) < tolf
    assert isclose(res.f, f(res.x))
    # stop maxiter
    maxiter = 3
    res = fzero_brent(f, 0.1, 1.0, maxiter=maxiter)
    assert not res.success
    assert "iterations" in res.message
    assert res.niter == maxiter
    assert isclose(res.f, f(res.x))
    # no change of sign in interval
    with pytest.raises(ValueError):
        _ = fzero_brent(f, 0.1, 0.2)
