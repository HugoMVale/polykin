# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.properties.thermal_conductivity import KLMX2_li, \
    KVPC_stiel_thodos, KVMX2_herning
import numpy as np


def test_KLMX2_li():
    "Example 36, p. 2-370, Perry's"
    x1 = 0.302
    M1 = 74.123
    M2 = 32.04
    w1 = x1*M1/(x1*M1 + (1 - x1)*M2)
    w = np.array([w1, 1 - w1])
    rho = np.array([0.713, 0.792])
    k = np.array([0.1383, 0.2069])
    km = KLMX2_li(w, k, rho)
    assert np.isclose(km, 0.167, rtol=1e-2)


def test_KVPC_stiel_thodos():
    "Example 10-3, p. 521, Reid-Prausnitz-Poling"
    Tc = 309.6
    Pc = 72.4e5
    Vc = 97.4  # cm3/mol
    Zc = 0.274
    M = 44.013e-3
    rhor = Vc/144
    kpc = KVPC_stiel_thodos(rhor, M, Tc, Pc, Zc)
    assert np.isclose(kpc, 1.78e-2, rtol=1e-2)


def test_KVMX2_herning():
    "Example 10-5, p. 534, Reid-Prausnitz-Poling"
    y1 = 0.25
    M1 = 78.114
    M2 = 39.948
    y = np.array([y1, 1 - y1])
    k = np.array([1.66e-2, 2.14e-2])
    M = np.array([M1, M2])
    k_mix = KVMX2_herning(y, k, M)
    assert np.isclose(k_mix, 1.92e-2, rtol=5e-2)
