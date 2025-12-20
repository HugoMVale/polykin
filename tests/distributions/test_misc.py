# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

import numpy as np
import pytest
from numpy import allclose, isclose

from polykin.distributions import (
    Flory,
    convert_polymer_standards,
    convolve_moments,
    convolve_moments_self,
    reconstruct_Laguerre,
)


def test_convolve_moments():
    q0 = 1.0
    q1 = q0 * 100.0
    q2 = q1 * 200.0
    p0, p1, p2 = convolve_moments([q0, q1, q2], [q0, q1, q2])
    assert isclose(p0 * p2 / p1**2, 1.5)
    # different lengths
    with pytest.raises(ValueError):
        convolve_moments([q0, q1], [q0, q1, q2])


def test_convolve_moments_self():

    def convolve_moments_self_iter(q, order):
        r = q
        for _ in range(order):
            r = convolve_moments(q, r)
        return r

    for order in range(1, 10):
        q0 = 1.0
        q1 = q0 * 100.0
        q2 = q1 * 200.0
        s = convolve_moments_self_iter([q0, q1, q2], order)
        p = convolve_moments_self([q0, q1, q2], order)
        assert allclose(s, p)

    pA = convolve_moments_self([q0, q1, q2], order=4)
    pB = convolve_moments_self([q0, q1, q2, q2 * 100], order=4)
    assert allclose(pA, pB[:3])


def test_convert_polymer_standards():
    a1 = 0.77  # PS in THF
    K1 = 6.82e-3  # PS in THF
    a2 = 0.69  # PMMA in THF
    K2 = 1.28e-2  # PMMA in THF
    # same polymer
    assert isclose(convert_polymer_standards(1.0, K1, K1, a1, a1), 1.0)
    # PS->PMMA
    M2 = convert_polymer_standards(100, K1, K2, a1, a2)
    assert isclose(M2, 85.68, rtol=1e-3)
    # vector
    M2 = convert_polymer_standards([20.0, 100.0, 200.0], K1, K2, a1, a2)
    assert allclose(M2, [15.87, 85.68, 177.07], rtol=1e-3)


def test_reconstruct_Laguerre():
    d = Flory(100)
    moments = [d._moment_length(i) for i in range(4)]
    drec = reconstruct_Laguerre(moments)
    n = np.arange(1, int(5 * d.DPz), 1, dtype=np.int_)
    pdf = drec(n)
    for i in range(len(moments)):
        assert isclose(np.dot(n**i, pdf), moments[i], rtol=1e-2)
