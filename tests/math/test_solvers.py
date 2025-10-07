# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024
from math import exp

import numpy as np
import pytest
from numba import njit
from numpy import isclose

from polykin.math.solvers import (fzero_brent, fzero_newton, fzero_secant,
                                  ode_rk)


def f(x):
    return 2*x**3 + 4*x**2 + x - 2


SOL = min(np.roots((2, 4, 1, -2)), key=lambda x: abs(x - 0.5))


def test_fzero_newton():
    # stop xtol
    xtol = 1e-9
    res = fzero_newton(f, 1.5, xtol=xtol, ftol=1e-100)
    assert res.success
    assert isclose(res.x, SOL, atol=xtol)
    # stop ftol
    ftol = 1e-10
    res = fzero_newton(f, 1.5, xtol=1e-100, ftol=ftol)
    assert res.success
    assert abs(res.f) < ftol
    # stop maxiter
    maxiter = 2
    res = fzero_newton(f, 1.5, maxiter=maxiter)
    assert not res.success
    assert res.niter == maxiter


def test_fzero_secant():
    # stop xtol
    xtol = 1e-9
    res = fzero_secant(f, 1.5, 1.4, xtol=xtol, ftol=1e-100)
    assert res.success
    assert isclose(res.x, SOL, atol=xtol)
    # stop ftol
    ftol = 1e-10
    res = fzero_secant(f, 1.5, 1.4, xtol=1e-100, ftol=ftol)
    assert res.success
    assert abs(res.f) < ftol
    # stop maxiter
    maxiter = 2
    res = fzero_secant(f, 1.5, 1.4, maxiter=maxiter)
    assert not res.success
    assert res.niter == maxiter


def test_fzero_brent():
    # stop xtol
    xtol = 1e-9
    res = fzero_brent(f, 0.1, 1.0, xtol=xtol, ftol=1e-100)
    assert res.success
    assert isclose(res.x, SOL, atol=xtol)
    # stop ftol
    ftol = 1e-10
    res = fzero_brent(f, 0.1, 1.0, xtol=1e-100, ftol=ftol)
    assert res.success
    assert abs(res.f) < ftol
    # stop maxiter
    maxiter = 2
    res = fzero_brent(f, 0.1, 1.0, maxiter=maxiter)
    assert not res.success
    assert res.niter == maxiter
    # no change of sign in interval
    with pytest.raises(ValueError):
        _ = fzero_brent(f, 0.1, 0.2)


def test_ode_rk():

    t0 = 0.0
    y0 = 1.0
    tf = 4.0

    @njit
    def ydot(t, y):
        return y + t**2

    def ysol(t):
        "Exact analytical solution."
        c = 3.
        return c*exp(t) - t**2 - 2*t - 2

    for h, order in zip([1e-5, 1e-2, 1e-1], [1, 2, 4]):
        yf = ode_rk(ydot, t0, tf, y0, h, order)  # type: ignore
        assert isclose(yf, ysol(tf), rtol=1e-4)

    with pytest.raises(ValueError):
        _ = ode_rk(ydot, 0., 1., 1., 1e-5, 5)  # type: ignore
