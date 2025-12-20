# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
import pytest
from numpy import isclose

from polykin.math.roots import root_brent, root_newton, root_secant


def f1(x):
    """Test for scalar root-finding methods."""
    return 2 * x**3 + 4 * x**2 + x - 2


f1.sol = min(np.roots((2, 4, 1, -2)), key=lambda x: abs(x - 0.5))


def f2(x):
    """Nasty test for scalar root-finding methods."""
    return np.exp(x - 1) * (x - 1) ** 2


f2.sol = 1.0


def test_root_newton():
    # stop tolx
    tolx = 1e-9
    res = root_newton(f1, 1.5, tolx=tolx, tolf=0.0)
    assert res.success
    assert "tolx" in res.message
    assert isclose(res.x, f1.sol, atol=tolx)
    assert isclose(res.f, f1(res.x))
    assert res.nfeval == res.niter
    # stop tolf
    tolf = 1e-10
    res = root_newton(f1, 1.5, tolx=0.0, tolf=tolf)
    assert res.success
    assert "tolf" in res.message
    assert abs(res.f) <= tolf
    assert isclose(res.f, f1(res.x))
    # stop maxiter
    maxiter = 3
    res = root_newton(f1, 1.5, maxiter=maxiter)
    assert not res.success
    assert "iterations" in res.message
    assert res.niter == maxiter
    assert isclose(res.f, f1(res.x))
    # nasty test
    res = root_newton(f2, 0.5, tolx=1e-10, tolf=1e-10)
    assert res.success
    assert isclose(res.x, f2.sol, atol=1e-10)
    # devirative is zero at the root
    res = root_newton(f2, 0.0, tolx=0, tolf=0, maxiter=100)
    assert not res.success
    assert "derivative" in res.message


def test_root_secant():
    # stop tolx
    tolx = 1e-9
    res = root_secant(f1, 1.5, 1.4, tolx=tolx, tolf=0.0)
    assert res.success
    assert "tolx" in res.message
    assert isclose(res.x, f1.sol, atol=tolx)
    assert isclose(res.f, f1(res.x))
    assert res.nfeval == res.niter + 2
    # stop tolf
    tolf = 1e-10
    res = root_secant(f1, 1.5, 1.4, tolx=0.0, tolf=tolf)
    assert res.success
    assert "tolf" in res.message
    assert abs(res.f) <= tolf
    assert isclose(res.f, f1(res.x))
    # stop maxiter
    maxiter = 3
    res = root_secant(f1, 1.5, 1.4, maxiter=maxiter)
    assert not res.success
    assert "iterations" in res.message
    assert res.niter == maxiter
    assert isclose(res.f, f1(res.x))
    # nasty test
    res = root_secant(f2, 0.5, 0.6, tolx=1e-10, tolf=1e-10)
    assert res.success
    assert isclose(res.x, f2.sol, atol=1e-10)
    # derivative is zero at the root
    res = root_secant(f2, 0.0, 0.1, tolx=0, tolf=0, maxiter=100)
    assert not res.success
    assert "zero slope" in res.message


def test_root_brent():
    # stop tolx
    tolx = 1e-9
    res = root_brent(f1, 0.1, 1.0, tolx=tolx, tolf=0.0)
    assert res.success
    assert "tolx" in res.message
    assert isclose(res.x, f1.sol, atol=tolx)
    assert isclose(res.f, f1(res.x))
    # stop tolf
    tolf = 1e-10
    res = root_brent(f1, 0.1, 1.0, tolx=0.0, tolf=tolf)
    assert res.success
    assert "tolf" in res.message
    assert abs(res.f) < tolf
    assert isclose(res.f, f1(res.x))
    # stop maxiter
    maxiter = 3
    res = root_brent(f1, 0.1, 1.0, maxiter=maxiter)
    assert not res.success
    assert "iterations" in res.message
    assert res.niter == maxiter
    assert isclose(res.f, f1(res.x))
    # a is solution
    res = root_brent(f1, f1.sol, 1.0)
    assert res.success
    assert res.niter == 0
    # b is solution
    res = root_brent(f1, 0.0, f1.sol)
    assert res.success
    assert res.niter == 0
    # root not bracketed
    with pytest.raises(ValueError):
        _ = root_brent(f1, 0.1, 0.2)
