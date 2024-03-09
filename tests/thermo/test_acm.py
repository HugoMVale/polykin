# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import isclose

from polykin.thermo.acm import IdealSolution, NRTL


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
