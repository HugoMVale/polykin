# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import isclose
from scipy.constants import gas_constant as R

from polykin.thermo.acm import (NRTL, UNIQUAC, FloryHuggins, FloryHuggins2_a,
                                FloryHuggins_a, IdealSolution)


def test_IdealSolution():
    m = IdealSolution(2)
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
    # activity formula <> DGmix
    x = np.array([0.3, 0.7])
    Dgmix = m.Dgmix(T, x)
    a = m.a(T, x)
    assert isclose(Dgmix, R*T*np.sum(x*np.log(a)))


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
    # activity formula <> DGmix
    Dgmix = m.Dgmix(T, x)
    a = m.a(T, x)
    assert isclose(Dgmix, R*T*np.sum(x*np.log(a)))


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


def test_FloryHuggins2_a():
    chi = 0.29
    gamma = FloryHuggins2_a(np.array([0., 0.25, 1.]), 1e10, chi)
    assert np.all(isclose(gamma, [0., 0.623, 1.], rtol=1e-2))


def test_FloryHuggins_a():
    # binary system
    chi = 0.4
    m = 10.
    phi1 = 0.25
    a1 = FloryHuggins2_a(phi1, m, chi)
    a = FloryHuggins_a(np.array([phi1, 1 - phi1]),
                       np.array([1, m]),
                       np.array([[0, chi], [chi, 0]]))
    assert isclose(a1, a[0])
    # ternary
    chi = 0.4
    m = 10.
    phi1 = 0.25
    a2 = FloryHuggins_a(np.array([phi1, 1 - phi1]),
                        np.array([1, m]),
                        np.array([[0, chi], [chi, 0]]))
    a3 = FloryHuggins_a(np.array([phi1/2, phi1/2, 1 - phi1]),
                        np.array([1, 1, m]),
                        np.array([[0, 0, chi], [0, 0, chi], [chi, chi, 0]]))
    assert np.all(isclose(a2, [a3[0:2].sum(), a3[-1]]))


def test_FloryHuggins():
    N = 2
    T = 298.
    phi1 = 0.3
    phi = np.array([phi1, 1-phi1])
    # ideal solution
    X = np.zeros((N, N))
    m = np.array([1, 1])
    fh = FloryHuggins(N, a=X)
    id = IdealSolution(N)
    assert isclose(fh.Dhmix(T, phi, m), 0.)
    assert isclose(fh.Dsmix(T, phi, m), id.Dsmix(T, phi))
    assert isclose(fh.Dgmix(T, phi, m), id.Dgmix(T, phi))
    # regular solution (DSmix=0, DHmix>0)
    chi = 0.4
    X = np.array([[0, chi], [chi, 0]])
    m = np.array([1, 1])
    fh = FloryHuggins(N, b=X*T)
    assert fh.Dhmix(T, phi, m) > 0
    assert isclose(fh.Dsmix(T, phi, m), id.Dsmix(T, phi))
    # activity formula <> DGmix
    chi = 0.3
    X = np.array([[0, chi], [chi, 0]])
    m = np.array([1, 13])
    fh = FloryHuggins(N, b=X*T)
    Dgmix = fh.Dgmix(T, phi, m)
    a = fh.a(T, phi, m)
    assert isclose(Dgmix, R*T*np.sum(phi/m*np.log(a)))
