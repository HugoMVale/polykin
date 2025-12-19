# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import pytest
from numba import njit
from numpy import exp, isclose

from polykin.math.derivatives.ode import ode_rk


def test_ode_rk():

    t0 = 0.0
    y0 = 1.0
    tf = 4.0

    @njit
    def ydot(t, y):
        return y + t**2

    def ysol(t):
        """Exact analytical solution."""
        c = 3.0
        return c * exp(t) - t**2 - 2 * t - 2

    for h, order in zip([1e-5, 1e-2, 1e-1], [1, 2, 4]):
        yf = ode_rk(ydot, t0, tf, y0, h, order)  # type: ignore
        assert isclose(yf, ysol(tf), rtol=1e-4)

    with pytest.raises(ValueError):
        _ = ode_rk(ydot, 0.0, 1.0, 1.0, 1e-5, 5)  # type: ignore
