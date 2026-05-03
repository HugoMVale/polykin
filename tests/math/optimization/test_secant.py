# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

import numpy as np
from numpy import isclose

from polykin.math.optimization import fmin_secant

TEST_FUNCTIONS = {
    "quadratic": {
        "f": lambda x: (x - 2.0) ** 2,
        "x0": 0.0,
        "xmin": 2.0,
        "fmin": 0.0,
    },
    "scaled_quadratic": {
        "f": lambda x: 1e6 * (x - 1e-3) ** 2,
        "x0": -1.0,
        "xmin": 1e-3,
        "fmin": 0.0,
    },
    "shifted_parabola": {
        "f": lambda x: 3.0 * (x + 0.7) ** 2 + 2.0,
        "x0": -3.0,
        "xmin": -0.7,
        "fmin": 2.0,
    },
    "anisotropic_scaling": {
        "f": lambda x: 1e3 * (x - 0.25) ** 2 + 1e-6,
        "x0": -1.0,
        "xmin": 0.25,
        "fmin": 1e-6,
    },
    "mildly_skewed_quadratic": {
        "f": lambda x: (x - 1.0) ** 2 + 0.05 * x,
        "x0": -2.0,
        "xmin": 0.975,  # shifted minimum
        "fmin": None,
    },
    "narrow_convex_basin": {
        "f": lambda x: np.exp(20 * (x - 0.4) ** 2),
        "x0": 0.0,
        "xmin": 0.4,
        "fmin": 1.0,
    },
}


def callback(niter, x, fx, dfx):
    print(f"Current point: x={x}, f(x)={fx}, df(x)={dfx}, iteration {niter}")
    return (niter >= 1, True)


def test_fmin_secant():
    for name, data in TEST_FUNCTIONS.items():
        for diff_scheme in ["centered", "complex"]:
            f = data["f"]
            x0 = data["x0"]
            xb = x0 + 0.01
            tolx = 1e-6
            tolg = 1e-7
            res = fmin_secant(f, x0, xb, tolx=tolx, tolg=tolg, diff_scheme=diff_scheme)

            assert res.success, f"Optimization failed for {name}: {res.message}"
            assert isclose(res.x, data["xmin"], atol=2 * tolx), (
                f"Incorrect minimum for {name}: x={res.x}, f(x)={res.f}"
            )
            # with callback
            res = fmin_secant(f, x0, xb, tolx=tolx, callback=callback)
            assert "callback" in res.message.lower()
            assert res.success
            assert res.niter == 1
