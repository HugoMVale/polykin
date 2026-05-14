# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

import numpy as np
from numpy import allclose

from polykin.math.optimization import fmin_qnewton

from ._tester import TEST_FUNCTIONS_MULTIVAR


def test_fmin_qnewton_basic():
    ATOL = 1e-5
    for fname in ["sphere", "ellipsoid"]:
        test = TEST_FUNCTIONS_MULTIVAR[fname]
        f = test["function"]
        grad = test["gradient"]
        hess = test["hessian"]

        for N in [2, 3, 5, 10]:
            x0 = test["initial_point"](N)
            xopt = test["global_minimizer"](N)
            for global_method in [None, "line-search", "dogleg"]:
                if global_method is None and N > 3:
                    continue
                for bfgs_update in [None, False, True]:
                    for bfgs_method in [None, "factored", "unfactored", "inverse"]:
                        for hess0 in [None, np.eye(N)]:
                            # No derivative information
                            res = fmin_qnewton(
                                f,
                                x0,
                                global_method=global_method,
                                bfgs_update=bfgs_update,
                                bfgs_method=bfgs_method,
                                H0=hess0,
                            )
                            assert allclose(res.x, xopt, atol=ATOL)
                            # With gradient
                            res = fmin_qnewton(
                                f,
                                x0,
                                global_method=global_method,
                                bfgs_update=bfgs_update,
                                bfgs_method=bfgs_method,
                                H0=hess0,
                                grad=grad,
                            )
                            assert allclose(res.x, xopt, atol=ATOL)
                            # With gradient and hessian
                            res = fmin_qnewton(
                                f,
                                x0,
                                global_method=global_method,
                                bfgs_update=bfgs_update,
                                bfgs_method=bfgs_method,
                                H0=hess0,
                                grad=grad,
                                hess=hess,
                            )
                            assert allclose(res.x, xopt, atol=ATOL)


def test_fmin_qnewton_advanced():
    ATOL = 1e-5
    for fname in ["zakharov"]:
        test = TEST_FUNCTIONS_MULTIVAR[fname]
        f = test["function"]
        grad = test["gradient"]
        hess = test["hessian"]
        for N in [2, 3, 5]:
            x0 = test["initial_point"](N)
            xopt = test["global_minimizer"](N)
            for global_method in ["line-search", "dogleg"]:
                for bfgs_update in [None, False, True]:
                    for bfgs_method in [None, "factored", "unfactored", "inverse"]:
                        for hess0 in [None, np.eye(N)]:
                            # No derivative information
                            res = fmin_qnewton(
                                f,
                                x0,
                                global_method=global_method,
                                bfgs_update=bfgs_update,
                                bfgs_method=bfgs_method,
                                H0=hess0,
                            )
                            assert allclose(res.x, xopt, atol=ATOL)
                            # With gradient
                            res = fmin_qnewton(
                                f,
                                x0,
                                global_method=global_method,
                                bfgs_update=bfgs_update,
                                bfgs_method=bfgs_method,
                                H0=hess0,
                                grad=grad,
                            )
                            assert allclose(res.x, xopt, atol=ATOL)
                            # With gradient and hessian
                            res = fmin_qnewton(
                                f,
                                x0,
                                global_method=global_method,
                                bfgs_update=bfgs_update,
                                bfgs_method=bfgs_method,
                                H0=hess0,
                                grad=grad,
                                hess=hess,
                            )
                            assert allclose(res.x, xopt, atol=ATOL)
