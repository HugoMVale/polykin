# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.properties.viscosity import MUVMX2_herning, MUVPC_jossi, \
    MUV_lucas, MUVMX_lucas, MULMX2_perry

import numpy as np


def test_MUVMX2_herning():
    "Example 9-6, p. 411"
    y2 = 0.303
    y = np.array([1-y2, y2])
    mu = np.array([109.4, 72.74])
    M = np.array([16.043, 58.124])
    mu_mix = MUVMX2_herning(y, mu, M)
    assert np.isclose(mu_mix, 92.82, rtol=1e-3)


def test_MUVPC_jossi():
    "Example 9-11, p. 425"
    Tc = 408.2
    Pc = 36.5e5
    rhor = 263/243.8
    M = 58.12e-3
    mu_pc = MUVPC_jossi(rhor, M, Tc, Pc)
    assert np.isclose(mu_pc, (273-120)*1e-7, rtol=2e-2)


def test_MUV_lucas():
    mu = MUV_lucas(T=420., P=300e5, M=17.03e-3,
                   Tc=405.5, Pc=113.5e5, Zc=0.244, dm=1.47)
    assert np.isclose(mu, 601e-7, rtol=1e-2)


def test_MUVMX_lucas():
    T = 420.
    P = 300e5
    M1 = 17.03e-3
    Tc1 = 405.5
    Pc1 = 113.5e5
    Zc1 = 0.244
    dm1 = 1.47
    y = np.array([1., 0.])
    M = np.array([M1, 123.])
    Tc = np.array([Tc1, 123.])
    Pc = np.array([Pc1, 123.])
    Zc = np.array([Zc1, 0.2])
    dm = np.array([dm1, 123.])
    mu1 = MUV_lucas(T=420., P=300e5, M=17.03e-3,
                    Tc=405.5, Pc=113.5e5, Zc=0.244, dm=1.47)
    mu_mix = MUVMX_lucas(T, P, y, M, Tc, Pc, Zc, dm)
    assert np.isclose(mu1, mu_mix)


def test_MULMX2_perry():
    x = np.array([0.5, 0.5])
    mu = np.array([1., 10])
    mu_mix = MULMX2_perry(x, mu, True)
    assert np.isclose(mu_mix, 5., atol=2.)
    mu_mix = MULMX2_perry(x, mu, False)
    assert np.isclose(mu_mix, 5., atol=2.)