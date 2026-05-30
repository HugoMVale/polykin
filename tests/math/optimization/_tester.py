"""Test functions commonly used in optimization benchmarks, with their gradients and
Hessians.
"""

import numpy as np


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


TEST_FUNCTIONS_MULTIVAR = {
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
