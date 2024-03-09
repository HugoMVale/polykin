# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import isclose

from polykin.thermo.acm import IdealSolution, NRTL, UNIQUAC


def test_IdealSolution():
    m = IdealSolution()
    T = 298.
    x = np.array([0.5, 0.5])
    assert isclose(m.gE(T, x), 0.)
    assert np.all(isclose(m.gamma(T, x), [1., 1.]))


def test_NRTL():
    a = np.array([[0, -0.178], [1.963, 0]])
    c = np.array([[0, 0.2974], [.2974, 0]])
    m = NRTL(2, a=a, c=c)
    T = 298.
    x = np.array([0.252, 0.748])
    # assert isclose(m.gE(T, x), 0.)
    assert np.all(isclose(m.gamma(T, x), [1.93631838, 1.15376097], rtol=1e-6))


def test_UNIQUAC():
    N = 3
    r = np.array([1.87, 3.19, 5.17])
    q = np.array([1.72, 2.4, 4.4])
    b = np.array(
        [[0.0, -60.28, -23.71], [-89.57, 0.0, 135.9], [-545.8, -245.4, 0.0]])
    m = UNIQUAC(N, q, r, b=b)
    x = np.array([.1311, .0330, .8359])
    T = 45 + 273.15
    # assert isclose(m.gE(T, x), 0.)
    assert np.all(isclose(m.gamma(T, x), [7.15, 1.25, 1.06], rtol=2e-3))
