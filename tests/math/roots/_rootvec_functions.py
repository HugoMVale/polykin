"""Benchmark functions for vector root-finding tests."""

import numpy as np


def f_powell_singular(x):
    n = x.size
    f = np.zeros(n)
    for k in range(n // 4):
        f[4 * k] = x[4 * k] - 10 * x[4 * k + 1]
        f[4 * k + 1] = np.sqrt(5) * (x[4 * k + 2] - x[4 * k + 3])
        f[4 * k + 2] = (x[4 * k + 1] - 2 * x[4 * k + 2]) ** 2
        f[4 * k + 3] = np.sqrt(10) * (x[4 * k] - x[4 * k + 3]) ** 2
    return f


def f_rosenbrock(x):
    n = x.size
    f = np.zeros(n)
    for k in range(n // 2):
        f[2 * k] = 1 - x[2 * k]
        f[2 * k + 1] = 10 * (x[2 * k + 1] - x[2 * k] ** 2)
    return f


def f_trignometric(x):
    n = x.size
    tsum = np.sum(np.cos(x))
    f = np.zeros(n)
    for k in range(n):
        f[k] = n - tsum + (k + 1) * (1 - np.cos(x[k])) - np.sin(x[k])
    return f


def f_case10(x):
    x1, x2 = x
    f1 = -13 + x1 + ((-x2 + 5) * x2 - 2) * x2
    f2 = -29 + x1 + ((x2 + 1) * x2 - 14) * x2
    return np.array([f1, f2])


def f_example65(x):
    x1, x2 = x
    f1 = x1**2 + x2**2 - 2
    f2 = np.exp(x1 - 1) + x2**3 - 2
    return np.array([f1, f2])


def jac_example65(x):
    x1, x2 = x
    return np.array([[2 * x1, 2 * x2], [np.exp(x1 - 1), 3 * x2**2]])


TEST_FUNCTIONS = {
    "powell_singular": {
        "f": f_powell_singular,
        "description": "Extended Powell singular function. Dennis & Schnabel (1996).",
        "x0": np.array([3.0, -1.0, 0.0, 1.0]),
        "xs": np.zeros(4),
    },
    "rosenbrock": {
        "f": f_rosenbrock,
        "description": "Extended Rosenbrock function. Dennis & Schnabel (1996).",
        "x0": np.array([-1.2, 1.0, -1.2, 1.0]),
        "xs": np.ones(4),
    },
    "trignometric": {
        "f": f_trignometric,
        "description": "Trigonometric function. Dennis & Schnabel (1996); Moré, Garbow & Hillstrom (1981), problem 26.",  # noqa: E501
        "x0": np.full(10, 0.01),
        "xs": np.zeros(10),
    },
    "case10": {
        "f": f_case10,
        "description": "Case 10 of Broyden (1965). Very tough.",
        "x0": np.array([15.0, -2.0]),
        "xs": np.array([5.0, 4.0]),
    },
    "example65": {
        "f": f_example65,
        "description": "Example 6.5 of Dennis & Schnabel (1996).",
        "x0": np.array([2.0, 0.5]),
        "xs": np.array([1.0, 1.0]),
    },
}
