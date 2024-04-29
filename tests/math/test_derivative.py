# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import exp, isclose

from polykin.math import derivative_centered, derivative_complex, hessian2


def fnc1(x): return 2*exp(x)


def test_derivative_centered():
    df, fx = derivative_centered(fnc1, 2.)
    assert isclose(df, fx)
    df, fx = derivative_centered(fnc1, 2., h=1e-5)
    assert isclose(df, fnc1(2.))


def test_derivative_complex():
    df, fx = derivative_complex(fnc1, 2)
    assert isclose(df, fx)


def test_hessian2():
    def fnc2(x): return x[0]**2 * x[1]**3
    x0 = 3.
    x1 = -2.
    H = hessian2(fnc2, (x0, x1))
    assert np.all(
        isclose(H, np.array([[2*x1**3, 6*x0*x1**2], [6*x0*x1**2, 6*x1*x0**2]]), rtol=1e-6))
    H = hessian2(fnc2, (x0, x1), h=1e-2)
    assert np.all(
        isclose(H, np.array([[2*x1**3, 6*x0*x1**2], [6*x0*x1**2, 6*x1*x0**2]]), rtol=1e-4))
