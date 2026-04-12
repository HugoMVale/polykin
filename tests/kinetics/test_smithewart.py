# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import isclose, sqrt

from polykin.kinetics.emulsion.smithewart import (
    SmithEwart_qssa_pdf,
    compartmentalization_factor,
    n2bar,
    nbar_Li_Brooks,
    nbar_Stockmayer_OToole,
    nbar_Ugelstad,
)

ALPHA_M_PAIRS = [(0.0, 1e-5), (1e-6, 1e-4), (1e0, 1e0), (1.0, 0.0), (1e1, 1e0)]


def test_nbar_Stockmayer_OToole():
    assert isclose(nbar_Stockmayer_OToole(alpha=1e-3, m=0.0), 0.5, rtol=1e-3)
    assert isclose(nbar_Stockmayer_OToole(alpha=0, m=1e-3), 0.0, atol=1e-10)
    assert isclose(nbar_Stockmayer_OToole(alpha=1e2, m=0.0), sqrt(1e2 / 2), rtol=5e-2)


def test_nbar_Li_Brooks():
    for alpha, m in ALPHA_M_PAIRS:
        nbar_ref = nbar_Stockmayer_OToole(alpha, m)
        nbar = nbar_Li_Brooks(alpha, m)
        assert isclose(nbar_ref, nbar, rtol=5e-2)


def test_nbar_Ugelstad():
    for alpha, m in ALPHA_M_PAIRS:
        nbar_ref = nbar_Stockmayer_OToole(alpha, m)
        nbar = nbar_Ugelstad(alpha, m)
        assert isclose(nbar_ref, nbar, rtol=1e-5)


def test_n2bar():
    assert isclose(n2bar(alpha=0.0, m=1e-6), 0.0)
    alpha = 1e4
    m = 0.0
    n2avg = n2bar(alpha, m)
    navg = nbar_Stockmayer_OToole(alpha, m)
    assert isclose(n2avg, navg**2, rtol=0.1)
    for alpha, m in ALPHA_M_PAIRS[1:]:
        n2avg = n2bar(alpha, m)
        n = np.arange(0, max(10, 2 * int(n2avg)))
        p = SmithEwart_qssa_pdf(n, alpha, m)
        n2avg_ref = np.dot(p, n**2)
        assert isclose(n2avg, n2avg_ref, rtol=1e-3)


def test_compartmentalization_factor():
    Df = compartmentalization_factor(alpha=1e5, m=1e-6)
    assert isclose(Df, 1.0, rtol=1e-2)


def test_SmithEwart_qssa_pdf():
    alpha = 1e-1
    m = 1e-1
    N = 10
    n = np.arange(0, N + 1)
    p = SmithEwart_qssa_pdf(n, alpha, m)
    assert isclose(p.sum(), 1.0, rtol=1e-5)
    nbar = (p * n).sum()
    nbar_ref = nbar_Stockmayer_OToole(alpha, m)
    assert isclose(nbar_ref, nbar, rtol=1e-5)
