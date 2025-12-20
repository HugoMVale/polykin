# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

import numpy as np
from numpy import allclose

from polykin.math.roots import rootvec_qnewton

# %% Test problems


def f_powell_singular(x):
    """Extended Powell singular function. Dennis & Schnabel (1996)."""
    n = x.size
    f = np.zeros(n)
    for k in range(n // 4):
        f[4 * k] = x[4 * k] - 10 * x[4 * k + 1]
        f[4 * k + 1] = np.sqrt(5) * (x[4 * k + 2] - x[4 * k + 3])
        f[4 * k + 2] = (x[4 * k + 1] - 2 * x[4 * k + 2]) ** 2
        f[4 * k + 3] = np.sqrt(10) * (x[4 * k] - x[4 * k + 3]) ** 2
    return f


f_powell_singular.x0 = np.array([3.0, -1.0, 0.0, 1.0])
f_powell_singular.xs = np.zeros_like(f_powell_singular.x0)


def f_rosenbrock(x):
    """Extended Rosenbrock function. Dennis & Schnabel (1996)."""
    n = x.size
    f = np.zeros(n)
    for k in range(n // 2):
        f[2 * k] = 1 - x[2 * k]
        f[2 * k + 1] = 10 * (x[2 * k + 1] - x[2 * k] ** 2)
    return f


f_rosenbrock.x0 = np.array([-1.2, 1.0, -1.2, 1.0])
f_rosenbrock.xs = np.ones_like(f_rosenbrock.x0)


def f_trignometric(x):
    """Trignometric function. Dennis & Schnabel (1996)."""
    n = x.size
    f = np.zeros(n)
    for k in range(n):
        tsum = 0
        for j in range(n):
            tsum += np.cos(x[j]) + k * (1 - np.cos(x[k])) - np.sin(x[k])
        f[k] = n - tsum
    return f


f_trignometric.x0 = np.ones(10) / 10
f_trignometric.xs = np.zeros_like(f_trignometric.x0)


def f_case10(x):
    """Case 10 of Broyden (1965). Very tough."""
    x1, x2 = x
    f1 = -13 + x1 + ((-x2 + 5) * x2 - 2) * x2
    f2 = -29 + x1 + ((x2 + 1) * x2 - 14) * x2
    return np.array([f1, f2])


f_case10.x0 = np.array([15.0, -2.0])
f_case10.xs = np.array([5.0, 4.0])


def f_example65(x):
    """Example 6.5. Dennis & Schnabel (1996)."""
    x1, x2 = x
    f1 = x1**2 + x2**2 - 2
    f2 = np.exp(x1 - 1) + x2**3 - 2
    return np.array([f1, f2])


def jac_example65(x):
    x1, x2 = x
    return np.array([[2 * x1, 2 * x2], [np.exp(x1 - 1), 3 * x2**2]])


f_example65.x0 = np.array([2.0, 0.5])
f_example65.xs = np.array([1.0, 1.0])

# %% Tests


def test_rootvec_qnewton():
    # With global method, all work even with Broyden's update
    funcs = [f_rosenbrock, f_powell_singular, f_trignometric, f_example65]
    for global_method in ["line-search", "dogleg"]:
        for broyden_update in [False, True]:
            for f in funcs:
                sol = rootvec_qnewton(
                    f,
                    f.x0,
                    tolf=1e-8,
                    global_method=global_method,
                    broyden_update=broyden_update,
                )
                assert sol.success
                if not f.__name__ == "f_powell_singular":
                    assert allclose(sol.x, f.xs)

    # Without global method, most work as well
    funcs = [f_rosenbrock, f_powell_singular, f_trignometric]
    for f in funcs:
        sol = rootvec_qnewton(f, f.x0, tolf=1e-8, global_method=None)
        assert sol.success
        if not f.__name__ == "f_powell_singular":
            assert allclose(sol.x, f.xs)

    # Easy ones also work with approximate jac0 and Broyden update
    f = f_example65
    sol = rootvec_qnewton(f, f.x0, broyden_update=True, jac0=np.eye(f.x0.size))
    assert sol.success

    # With analytic jacobian
    f = f_example65
    jac = jac_example65
    for broyden_update in [False, True]:
        sol = rootvec_qnewton(f, f.x0, jac=jac, broyden_update=broyden_update)
        assert sol.success
        assert allclose(sol.x, f.xs)

    # Initial guess is solution
    f = f_example65
    sol = rootvec_qnewton(f, f.xs)
    assert sol.success
    assert sol.niter == 0
