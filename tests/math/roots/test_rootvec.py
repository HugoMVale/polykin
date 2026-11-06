# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

import numpy as np
from numpy import allclose

from polykin.math.roots import rootvec_qnewton

from ._rootvec_functions import TEST_FUNCTIONS, jac_example65

# %% Tests


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
