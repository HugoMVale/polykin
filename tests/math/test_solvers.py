# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024
from math import exp

from numpy import isclose

from polykin.math.solvers import ode_rk2, root_newton, root_secant

from numba import njit


def f(x):
    return 2*x**3 + 4*x**2 + x - 2


def test_root_newton():
    # normal
    res = root_newton(f, 1.5, ftol=1e-100)
    assert res.success
    assert isclose(res.x, 0.54, atol=0.01)
    # stop maxiter
    res = root_newton(f, 1.5, maxiter=2)
    assert not res.success
    # stop ftol
    ftol = 1e-10
    res = root_newton(f, 1.5, xtol=1e-4, ftol=ftol)
    assert res.success
    assert not (abs(res.f) < ftol)
    res = root_newton(f, 1.5, xtol=1e-100, ftol=ftol)
    assert res.success
    assert abs(res.f) < ftol


def test_root_secant():
    # normal
    res = root_secant(f, 1.5, 1.4)
    assert res.success
    assert isclose(res.x, 0.54, atol=0.01)
    # stop maxiter
    res = root_secant(f, 1.5, 1.4, maxiter=2)
    assert not res.success
    # stop ftol
    ftol = 1e-10
    res = root_secant(f, 1.5, 1.4, xtol=1e-4, ftol=ftol)
    assert res.success
    assert not (abs(res.f) < ftol)
    res = root_secant(f, 1.5, 1.4, xtol=1e-100, ftol=ftol)
    assert res.success
    assert abs(res.f) < ftol


def test_ode_rk2():
    @njit
    def ydot(t, y):
        return y + t**2

    t0 = 0.
    y0 = 1.
    tf = 4.
    h = 1e-2
    yf = ode_rk2(t0, tf, y0, h, ydot)

    def ysol(t):
        c = 3.
        return c*exp(t) - t**2 - 2*t - 2

    assert isclose(yf, ysol(tf), rtol=1e-4)
