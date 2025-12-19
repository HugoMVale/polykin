# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

import numpy as np
from numpy import allclose, isclose

from polykin.distributions import WeibullNycanderGold_pdf


def test_WeibullNycanderGold_pdf():
    v = 10.0
    c = 2.0
    s = np.arange(4 * int(v))
    Ps = WeibullNycanderGold_pdf(s, v, c)
    assert isinstance(Ps, np.ndarray) and s.shape == Ps.shape
    assert isclose(Ps.sum(), 1.0)
    # examples from Gold (1957)
    r = 0.1
    p0 = 0.1174e-1
    res = _moments(p0, r)
    assert allclose(res, (1.350, 1.614), rtol=1e-3)
    r = 0.5
    p0 = 0.1353
    res = _moments(p0, r)
    assert allclose(res, (1.657, 2.094), rtol=1e-3)
    r = 0.5
    p0 = 3.355e-4
    res = _moments(p0, r)
    assert allclose(res, (4.501, 5.334), rtol=1e-3)
    r = 2.0
    p0 = 0.3679
    res = _moments(p0, r)
    assert allclose(res, (2.164, 2.849), rtol=1e-3)
    r = 10.0
    p0 = 0.4594
    res = _moments(p0, r)
    assert allclose(res, (5.388, 7.110), rtol=1e-3)
    r = 100.0
    p0 = 0.3642
    res = _moments(p0, r)
    assert allclose(res, (59.87, 74.36), rtol=1e-3)


def _moments(p0, r):
    v = -r * np.log(p0) - (r - 1) * (1 - p0)
    s = np.arange(1, 10 * int(v))
    Ps = WeibullNycanderGold_pdf(s, v, r)
    assert isinstance(Ps, np.ndarray)
    m0 = Ps.sum()
    m1 = np.dot(s, Ps)
    m2 = np.dot(s**2, Ps)
    xn = m1 / m0
    xw = m2 / m1
    return (xn, xw)
