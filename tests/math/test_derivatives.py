# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import allclose, exp, isclose

from polykin.math import (derivative_centered, derivative_complex, hessian2,
                          jacobian, scalex)


def fnc1(x): return 2*exp(x)


def fnc2(x): return x[0]**2 * x[1]**3


def fnc3(x):
    x1, x2 = x
    f1 = 0.5*np.cos(x1) + 0.1*x2 + 0.5
    f2 = np.sin(x2) - 0.2*x1 + 1.2
    return np.array([f1, f2])


def test_derivative_centered():
    df, fx = derivative_centered(fnc1, 2.)
    assert isclose(df, fx)
    df, fx = derivative_centered(fnc1, 2., h=1e-5)
    assert isclose(df, fnc1(2.))


def test_derivative_complex():
    df, fx = derivative_complex(fnc1, 2)
    assert isclose(df, fx)


def test_hessian2():
    x0 = 3.
    x1 = -2.
    H = hessian2(fnc2, (x0, x1))
    assert allclose(H, np.array(
        [[2*x1**3, 6*x0*x1**2], [6*x0*x1**2, 6*x1*x0**2]]), rtol=1e-6)
    H = hessian2(fnc2, (x0, x1), h=1e-2)
    assert allclose(H, np.array(
        [[2*x1**3, 6*x0*x1**2], [6*x0*x1**2, 6*x1*x0**2]]), rtol=1e-4)


def test_jacobian():
    x = np.array([3.0, -2.0])
    J = jacobian(fnc3, x)
    assert allclose(J, np.array([[-0.07055999, 0.1], [-0.2, -0.41614682]]),
                    rtol=1e-6)


def test_scalex():
    sx = scalex(np.array([0.0, 0.0]))
    assert allclose(sx, np.ones_like(sx))
    sx = scalex(np.array([0.0, -1e2]))
    assert allclose(sx, [1e-1, 1e-2])
    sx = scalex(np.array([0.0, 0.1, -1e2]))
    assert allclose(sx, [1e2, 1e1, 1e-2])
    sx = scalex(np.array([0.0, 1.0, 5.0]))
    assert allclose(sx, [10.0, 0.2, 0.2])
