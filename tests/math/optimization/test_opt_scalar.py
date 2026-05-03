# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

import numpy as np
from numpy import isclose

from polykin.math.optimization import fmin_brent

TEST_FUNCTIONS_SCALAR = {
    "quadratic": {
        "f": lambda x: (x - 2.0) ** 2,
        "xa": -5.0,
        "xb": 5.0,
        "xmin": 2.0,
        "fmin": 0.0,
    },
    "scaled_quadratic": {
        "f": lambda x: 1e6 * (x - 1e-3) ** 2,
        "xa": -1.0,
        "xb": 1.0,
        "xmin": 1e-3,
        "fmin": 0.0,
    },
    "quartic_flat": {
        "f": lambda x: (x - 1.0) ** 4,
        "xa": -2.0,
        "xb": 3.0,
        "xmin": 1.0,
        "fmin": 0.0,
    },
    "absolute_value": {
        "f": lambda x: abs(x - 0.5),
        "xa": -1.0,
        "xb": 2.0,
        "xmin": 0.5,
        "fmin": 0.0,
    },
    "multi_minima": {
        "f": lambda x: np.sin(5 * x) + (x - 1) ** 2,
        "xa": -2.0,
        "xb": 3.0,
        "xmin": 0.9467389984,
        "fmin": -0.996,  # approximate
    },
    "degenerate_parabola": {
        "f": lambda x: (x - 1) ** 2 + 1e-12 * x,
        "xa": 0.0,
        "xb": 2.0,
        "xmin": 1.0 - 5e-13,  # exact minimizer ≈ 1 - ε/2
        "fmin": 1e-12 * (1.0 - 5e-13),
    },
    "sharp_minimum": {
        "f": lambda x: np.exp(50 * (x - 0.3) ** 2),
        "xa": 0.0,
        "xb": 1.0,
        "xmin": 0.3,
        "fmin": 1.0,
    },
    "boundary_minimum": {
        "f": lambda x: (x + 2) ** 2,
        "xa": -2.0,
        "xb": 5.0,
        "xmin": -2.0,
        "fmin": 0.0,
    },
}


def callback(niter, x, fx):
    print(f"Current point: x={x}, f(x)={fx}, iteration {niter}")
    return (niter >= 5, True)


def test_fmin_brent():
    for name, data in TEST_FUNCTIONS_SCALAR.items():
        f = data["f"]
        xa = data["xa"]
        xb = data["xb"]
        tolx = 1e-6
        res = fmin_brent(f, xa, xb, tolx=tolx)

        assert res.success, f"Optimization failed for {name}: {res.message}"
        assert isclose(res.x, data["xmin"], atol=2 * tolx), (
            f"Incorrect minimum for {name}: x={res.x}, f(x)={res.f}"
        )
        # with callback
        res = fmin_brent(f, xa, xb, tolx=tolx, callback=callback)
        assert "callback" in res.message.lower()
        assert res.success
        assert res.niter == 5
