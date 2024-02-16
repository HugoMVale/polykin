# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from scipy.constants import R

from polykin.properties.viscosity import (MULMX2_Perry, MUV_Lucas,
                                          MUVMX2_Herning_Zipperer, MUVMX_Lucas,
                                          MUVMXPC_Dean_Stiel, MUVPC_Jossi)


def test_MUVMX2_herning_zipperer():
    "Example 9-6, p. 411, Reid-Prausnitz-Poling."
    y2 = 0.303
    y = [1-y2, y2]
    mu = [109.4, 72.74]
    M = [16.043, 58.124]
    mu_mix = MUVMX2_Herning_Zipperer(y, mu, M)
    assert np.isclose(mu_mix, 92.82, rtol=1e-3)


def test_MUVPC_jossi():
    "Example 9-11, p. 425, Reid-Prausnitz-Poling."
    Tc = 408.2
    Pc = 36.5e5
    rhor = 263/243.8
    M = 58.12e-3
    mu_pc = MUVPC_Jossi(rhor, M, Tc, Pc)
    assert np.isclose(mu_pc, 1.53e-5, rtol=2e-2)


def test_MUVMXPC_dean_stiel_1():
    "Example 9-11, p. 425, Reid-Prausnitz-Poling."
    y = [1.]
    M = [58.12e-3]
    Tc = np.array([408.2])
    Pc = np.array([36.5e5])
    Zc = Pc*263e-6/(R*Tc)
    V = 243.8e-6
    mu_pc = MUVMXPC_Dean_Stiel(V, y, M, Tc, Pc, Zc)
    assert np.isclose(mu_pc, 1.53e-5, rtol=0.1)


def test_MUVMXPC_dean_stiel_2():
    "Example 23, p. 2-363, Perry's"
    y = [0.6, 0.4]
    M = [16.04e-3, 44.10e-3]
    Tc = np.array([-110.4, 96.7]) + 273.15
    Pc = [4.593e6, 4.246e6]
    Zc = [0.288, 0.281]
    V = 1/3.55e3
    mu_pc = MUVMXPC_Dean_Stiel(V, y, M, Tc, Pc, Zc)
    assert np.isclose(mu_pc, 3.7e-6, rtol=0.1)


def test_MUV_lucas():
    mu = MUV_Lucas(T=420., P=300e5, M=17.03e-3,
                   Tc=405.5, Pc=113.5e5, Zc=0.244, dm=1.47)
    assert np.isclose(mu, 601e-7, rtol=1e-2)
    mu1 = MUV_Lucas(T=299.9, P=299.9e5, M=100e-3,
                    Tc=300., Pc=300e5, Zc=0.28, dm=1.)
    mu2 = MUV_Lucas(T=300.1, P=300.1e5, M=100e-3,
                    Tc=300., Pc=300e5, Zc=0.28, dm=1.)
    assert np.isclose(mu1, mu2, rtol=1e-2)


def test_MUVMX_lucas():
    T = 420.
    P = 300e5
    M1 = 17.03e-3
    Tc1 = 405.5
    Pc1 = 113.5e5
    Zc1 = 0.244
    dm1 = 1.47
    y = [1., 0.]
    M = [M1, 123.]
    Tc = [Tc1, 123.]
    Pc = [Pc1, 123.]
    Zc = [Zc1, 0.2]
    dm = [dm1, 123.]
    mu1 = MUV_Lucas(T=420., P=300e5, M=17.03e-3,
                    Tc=405.5, Pc=113.5e5, Zc=0.244, dm=1.47)
    mu_mix = MUVMX_Lucas(T, P, y, M, Tc, Pc, Zc, dm)
    assert np.isclose(mu1, mu_mix)


def test_MULMX2_perry():
    x = [0.5, 0.5]
    mu = [1., 10]
    mu_mix = MULMX2_Perry(x, mu, True)
    assert np.isclose(mu_mix, 5., atol=2.)
    mu_mix = MULMX2_Perry(x, mu, False)
    assert np.isclose(mu_mix, 5., atol=2.)
