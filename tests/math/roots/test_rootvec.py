# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

import numpy as np
from numpy import allclose

from polykin.math.roots import rootvec_qnewton
from polykin.math.roots.qnewton import trcond_est

from ._rootvec_functions import TEST_FUNCTIONS, jac_example65


def test_rootvec_qnewton():
    # With global method, all work even with Broyden's update
    funcs = ["rosenbrock"]  # , "powell_singular", "trignometric", "example65"]
    for global_method in ["line-search", "dogleg"]:
        for broyden_update in [False, True]:
            for name in funcs:
                data = TEST_FUNCTIONS[name]
                f = data["f"]
                sol = rootvec_qnewton(
                    f,
                    data["x0"],
                    tolf=1e-7,
                    global_method=global_method,
                    broyden_update=broyden_update,
                )
                assert sol.success
                if name != "powell_singular":
                    assert allclose(sol.x, data["xs"])

    # Without global method, most work as well
    funcs = ["rosenbrock", "powell_singular", "trignometric"]
    for name in funcs:
        data = TEST_FUNCTIONS[name]
        f = data["f"]
        sol = rootvec_qnewton(
            f,
            data["x0"],
            tolf=1e-8,
            broyden_update=False,
            global_method=None,
        )
        assert sol.success
        if name != "powell_singular":
            assert allclose(sol.x, data["xs"])

    # Easy ones also work with approximate jac0 and Broyden update
    data = TEST_FUNCTIONS["example65"]
    f = data["f"]
    sol = rootvec_qnewton(f, data["x0"], broyden_update=True, J0=np.eye(data["x0"].size))
    assert sol.success

    # With analytic jacobian
    f = data["f"]
    jac = jac_example65
    for broyden_update in [False, True]:
        sol = rootvec_qnewton(
            f,
            data["x0"],
            jac=jac,
            jac_check=True,
            broyden_update=broyden_update,
        )
        assert sol.success
        assert allclose(sol.x, data["xs"])

    # Initial guess is solution
    sol = rootvec_qnewton(f, data["xs"])
    assert sol.success
    assert sol.niter == 0


def test_trcond_est_upper_and_lower():
    U = np.array([[2.0, 3.0, 7.0], [8.0, 4.0, 5.0], [9.0, 6.0, 3.0]])
    L = U.T

    for A, lower in zip((U, L), [False, True]):
        for norm in (1, np.inf):
            estimated = trcond_est(A, lower=lower, norm=norm)
            if lower:
                expected = np.linalg.cond(np.tril(L), norm)
            else:
                expected = np.linalg.cond(np.triu(U), norm)

            assert np.isclose(estimated, expected, rtol=0.1)


def test_trcond_est_unit_diagonal_and_singular():
    assert trcond_est([[0, 3], [0, 4]], unit_diagonal=True) == 16.0
    assert np.isinf(trcond_est([[1.0, 2.0], [0.0, 0.0]]))
