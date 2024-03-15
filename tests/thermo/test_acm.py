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
    # water: 0
    # ethanol: 1
    N = 2
    a = np.zeros((N, N))
    b = np.zeros((N, N))
    a[0, 1] = 3.4578
    a[1, 0] = -0.8009
    b[0, 1] = -586.0809
    b[1, 0] = 246.18
    m = NRTL(N, a, b)
    T = 298.15
    assert np.all(isclose(m.gamma(T, np.array([0., 1.])),
                          [2.660317, 1.], rtol=1e-6))
    assert np.all(isclose(m.gamma(T, np.array([1., 0.])),
                          [1., 4.557085], rtol=1e-6))
    assert isclose(m.Dgmix(T, np.array([0.5, 0.5])), -0.98183e3, rtol=1e-4)


def test_UNIQUAC():
    N = 3
    r = np.array([1.87, 3.19, 5.17])
    q = np.array([1.72, 2.4, 4.4])
    b = np.array(
        [[0.0, -60.28, -23.71], [-89.57, 0.0, 135.9], [-545.8, -245.4, 0.0]])
    m = UNIQUAC(N, q, r, b=b)
    x = np.array([.1311, .0330, .8359])
    T = 45 + 273.15
    assert np.all(isclose(m.gamma(T, x), [7.15, 1.25, 1.06], rtol=2e-3))


def test_UNIQUAC_2():
    # water: 0
    # ethanol: 1
    N = 2
    a = np.zeros((N, N))
    b = np.zeros((N, N))
    a[0, 1] = -2.4936
    a[1, 0] = 2.0046
    b[0, 1] = 756.9477
    b[1, 0] = -728.9705
    r = np.array([0.92, 2.1054])
    q = np.array([1.3997, 1.9720])
    m = UNIQUAC(N, q, r, a, b)
    T = 298.15
    assert np.all(isclose(m.gamma(T, np.array([0., 1.])),
                          [2.698527, 1.], rtol=1e-2))
    assert np.all(isclose(m.gamma(T, np.array([1., 0.])),
                          [1., 4.604476], rtol=1e-2))
    assert isclose(m.Dgmix(T, np.array([0.5, 0.5])), -0.98972e3, rtol=1e-2)