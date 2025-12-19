# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import allclose, isclose

from polykin.math import (
    derivative_centered,
    derivative_complex,
    hessian2_centered,
    hessian_forward,
    jacobian_forward,
    scalex,
)

# %% Test functions


def fnc1(x):
    return 2 * np.exp(x)


def fnc2(x):
    return x[0] ** 2 * x[1] ** 3


def fnc3(x):
    x1, x2 = x
    f1 = 0.5 * np.cos(x1) + 0.1 * x2 + 0.5
    f2 = np.sin(x2) - 0.2 * x1 + 1.2
    return np.array([f1, f2])


# %% Tests


def test_derivative_centered():
    df, fx = derivative_centered(fnc1, 2.0)
    assert isclose(df, fx)
    df, fx = derivative_centered(fnc1, 2.0, h=1e-5)
    assert isclose(df, fnc1(2.0))


def test_derivative_complex():
    df, fx = derivative_complex(fnc1, 2)
    assert isclose(df, fx)


def test_hessian2_centered():
    x0 = 3.0
    x1 = -2.0
    H = hessian2_centered(fnc2, (x0, x1))
    assert allclose(
        H,
        np.array([[2 * x1**3, 6 * x0 * x1**2], [6 * x0 * x1**2, 6 * x1 * x0**2]]),
        rtol=1e-6,
    )
    H = hessian2_centered(fnc2, (x0, x1), h=1e-2)
    assert allclose(
        H,
        np.array([[2 * x1**3, 6 * x0 * x1**2], [6 * x0 * x1**2, 6 * x1 * x0**2]]),
        rtol=1e-4,
    )


def test_jacobian_forward():
    x = np.array([3.0, -2.0])
    J = jacobian_forward(fnc3, x)
    assert allclose(J, np.array([[-0.07055999, 0.1], [-0.2, -0.41614682]]), rtol=1e-6)


def test_hessian_forward():
    x0 = 3.0
    x1 = -2.0
    x = np.array([x0, x1])
    H = hessian_forward(fnc2, x)
    assert allclose(
        H,
        np.array([[2 * x1**3, 6 * x0 * x1**2], [6 * x0 * x1**2, 6 * x1 * x0**2]]),
        rtol=1e-5,
    )


def test_scalex():
    sclx = scalex(np.array([0.0, 0.0]))
    assert allclose(sclx, np.ones_like(sclx))
    sclx = scalex(np.array([0.0, -1e2]))
    assert allclose(sclx, [1e-1, 1e-2])
    sclx = scalex(np.array([0.0, 0.1, -1e2]))
    assert allclose(sclx, [1e2, 1e1, 1e-2])
    sclx = scalex(np.array([0.0, 1.0, 5.0]))
    assert allclose(sclx, [10.0, 0.2, 0.2])
