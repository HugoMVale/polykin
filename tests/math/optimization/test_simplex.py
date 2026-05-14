# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

from numpy import isclose

from polykin.math.optimization import fmin_nelder_mead

from ._tester import TEST_FUNCTIONS_MULTIVAR


def test_fmin_nelder_mead():
    for name, data in TEST_FUNCTIONS_MULTIVAR.items():
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
        res = fmin_nelder_mead(
            f,
            x0,
            tolx=tolx,
            callback=lambda niter, x, fx: (niter >= 3, True),
        )
        assert res.success
        assert "callback" in res.message
        assert res.niter == 3
