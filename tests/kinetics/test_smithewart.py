# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import isclose, sqrt

from polykin.kinetics.emulsion.smithewart import (
    SmithEwart_qssa_pdf,
    nbar_Li_Brooks,
    nbar_Stockmayer_OToole,
    nbar_Ugelstad,
)


def test_nbar_Stockmayer_OToole():
    assert isclose(nbar_Stockmayer_OToole(alpha=1e-3, m=0.0), 0.5, rtol=1e-3)
    assert isclose(nbar_Stockmayer_OToole(alpha=0, m=1e-3), 0.0, atol=1e-10)
    assert isclose(nbar_Stockmayer_OToole(alpha=1e2, m=0.0), sqrt(1e2 / 2), rtol=5e-2)


def test_nbar_Li_Brooks():
    for alpha, m in [(0.0, 1e-5), (1e-6, 1e-4), (1e0, 1e0), (1.0, 0.0), (1e1, 1e0)]:
        nbar_ref = nbar_Stockmayer_OToole(alpha, m)
        nbar = nbar_Li_Brooks(alpha, m)
        assert isclose(nbar_ref, nbar, rtol=5e-2)


def test_nbar_Ugelstad():
    for alpha, m in [(0.0, 1e-5), (1e-6, 1e-4), (1e0, 1e0), (1.0, 0.0), (1e1, 1e0)]:
        nbar_ref = nbar_Stockmayer_OToole(alpha, m)
        nbar = nbar_Ugelstad(alpha, m)
        assert isclose(nbar_ref, nbar, rtol=1e-5)


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
