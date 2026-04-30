# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

import numpy as np
from numpy import isclose

from polykin.math.optimization import fmin_nelder_mead, fmin_qnewton

# Test functions and helpers


def rosenbrock_grad(x):
    g = np.zeros_like(x)
    g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g


def rosenbrock_hess(x):
    n = len(x)
    H = np.zeros((n, n))
    # Diagonal elements
    H[np.arange(n - 1), np.arange(n - 1)] += -400.0 * (x[1:] - 3 * x[:-1] ** 2) + 2.0
    H[np.arange(1, n), np.arange(1, n)] += 200.0
    # Off-diagonal elements
    H[np.arange(n - 1), np.arange(1, n)] = -400.0 * x[:-1]
    H[np.arange(1, n), np.arange(n - 1)] = -400.0 * x[:-1]
    return H


def ellipsoid_coeffs(x):
    # Safe division to handle n=1 edge case
    n = len(x)
    denom = n - 1 if n > 1 else 1
    return (10**6) ** (np.arange(n) / denom)


def zakharov_u(x):
    # 0.5 * sum(i * x_i), using 1-based indexing for the coefficient
    return np.sum(0.5 * (np.arange(len(x)) + 1) * x)


def zakharov_grad(x):
    u = zakharov_u(x)
    scale = 0.5 * (np.arange(len(x)) + 1)
    return 2.0 * x + (2.0 * u + 4.0 * u**3) * scale


def zakharov_hess(x):
    n = len(x)
    u = zakharov_u(x)
    scale = 0.5 * (np.arange(n) + 1)
    return 2.0 * np.eye(n) + (2.0 + 12.0 * u**2) * np.outer(scale, scale)


# --- The Dictionary ---

TEST_FUNCTIONS_VECTOR = {
    "rosenbrock": {
        "function": lambda x: np.sum(
            100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2
        ),
        "gradient": rosenbrock_grad,
        "hessian": rosenbrock_hess,
        "global_minimum": 0.0,
        "global_minimizer": lambda n: np.ones(n),
        # classic challenging but not extreme start
        "initial_point": lambda n: np.full(n, -1.2),
        "properties": ["valley", "non-convex"],
    },
    "ellipsoid": {
        "function": lambda x: np.sum(ellipsoid_coeffs(x) * x**2),
        "gradient": lambda x: 2.0 * ellipsoid_coeffs(x) * x,
        "hessian": lambda x: np.diag(2.0 * ellipsoid_coeffs(x)),
        "global_minimum": 0.0,
        "global_minimizer": lambda n: np.zeros(n),
        # asymmetric start → exposes conditioning issues
        "initial_point": lambda n: np.linspace(1.0, 2.0, n),
        "properties": ["ill-conditioned", "convex"],
    },
    "sphere": {
        "function": lambda x: np.sum(x**2),
        "gradient": lambda x: 2.0 * x,
        "hessian": lambda x: 2.0 * np.eye(len(x)),
        "global_minimum": 0.0,
        "global_minimizer": lambda n: np.zeros(n),
        # Simple start to verify basic algorithm correctness
        "initial_point": lambda n: np.full(n, 5.0),
        "properties": ["convex", "well-conditioned", "separable"],
    },
    "zakharov": {
        "function": lambda x: np.sum(x**2) + zakharov_u(x) ** 2 + zakharov_u(x) ** 4,
        "gradient": zakharov_grad,
        "hessian": zakharov_hess,
        "global_minimum": 0.0,
        "global_minimizer": lambda n: np.zeros(n),
        # Initial point recommended by standard benchmark literature
        "initial_point": lambda n: np.full(n, 1.5),
        "properties": ["convex", "steep-walls"],
    },
}


def test_fmin_nelder_mead():
    for name, data in TEST_FUNCTIONS_VECTOR.items():
        N = 2  # test in 2D for simplicity, but should work in any dimension
        f = data["function"]
        x0 = data["initial_point"](N)
        tolx = 1e-6
        res = fmin_nelder_mead(f, x0, tolx=tolx)

        assert res.success, f"Optimization failed for {name}: {res.message}"
        assert isclose(res.f, data["global_minimum"], atol=2 * tolx), (
            f"Incorrect minimum for {name}: x={res.x}, f(x)={res.f}"
        )

        # with callback
        res = fmin_nelder_mead(f, x0, tolx=tolx, callback=lambda niter, x, fx: niter >= 3)
        assert res.success
        assert "callback" in res.message
        assert res.niter == 3
