# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import isclose

from polykin.kinetics import nbar_Li_Brooks, nbar_Stockmayer_OToole


def test_nbar_Stockmayer_OToole():
    alpha = 1e-3
    m = 0.0
    nbar = nbar_Stockmayer_OToole(alpha, m)
    assert isclose(nbar, 0.5, rtol=1e-3)


def test_nbar_Li_Brooks():
    alpha = 1e0
    m = 1e0
    nbar_SOT = nbar_Stockmayer_OToole(alpha, m)
    nbar_LB = nbar_Li_Brooks(alpha, m)
    assert isclose(nbar_SOT, nbar_LB, rtol=4e-2)
