# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.properties.thermal_conductivity import KLMX2_li, \
    KVPC_stiel_thodos, KVMXPC_stiel_thodos, KVMX2_wassilijewa

import numpy as np
from scipy.constants import R
import pytest


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
    Zc = 0.274
    M = 44.013e-3
    V = 144e-6
    kpc = KVPC_stiel_thodos(V, M, Tc, Pc, Zc)
    assert np.isclose(kpc, 1.78e-2, rtol=1e-2)


def test_KVPC_stiel_thodos_continuity():
    Tc = 1.
    Pc = 1.
    Zc = 1/R
    M = 1.
    dV = 0.001
    for V in [1/0.5, 1/2.]:
        k1 = KVPC_stiel_thodos(V-dV, M, Tc, Pc, Zc)
        k2 = KVPC_stiel_thodos(V+dV, M, Tc, Pc, Zc)
        assert np.isclose(k1, k2, rtol=2e-2)
    with pytest.raises(ValueError):
        _ = KVPC_stiel_thodos(1/2.81, M, Tc, Pc, Zc)


def test_KVMXPC_stiel_thodos():
    "Example 10-6, p. 537, Reid-Prausnitz-Poling"
    y1 = 0.755
    V = 159e-6
    y = np.array([y1, 1 - y1])
    Zc = np.array([0.288, 0.274])
    Pc = np.array([46.0e5, 73.8e5])
    Tc = np.array([190.4, 304.1])
    M = np.array([16.043e-3, 44.010e-3])
    w = np.array([0.011, 0.239])
    kpc = KVMXPC_stiel_thodos(V, y, M, Tc, Pc, Zc, w)
    assert np.isclose(kpc, 1.50e-2, rtol=1e-2)


def test_KVMX2_wassilijewa():
    "Example 10-5, p. 534, Reid-Prausnitz-Poling"
    y1 = 0.25
    M1 = 78.114
    M2 = 39.948
    y = np.array([y1, 1 - y1])
    k = np.array([1.66e-2, 2.14e-2])
    M = np.array([M1, M2])
    k_mix = KVMX2_wassilijewa(y, k, M)
    assert np.isclose(k_mix, 1.92e-2, rtol=5e-2)
