# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

import numpy as np
from numpy import allclose, isclose

from polykin.distributions import (Flory, convert_polymer_standards,
                                   reconstruct_Laguerre)


def test_convert_polymer_standards():
    a1 = 0.77      # PS in THF
    K1 = 6.82e-3   # PS in THF
    a2 = 0.69      # PMMA in THF
    K2 = 1.28e-2   # PMMA in THF
    # same polymer
    assert isclose(convert_polymer_standards(1.0, K1, K1, a1, a1), 1.0)
    # PS->PMMA
    M2 = convert_polymer_standards(100, K1, K2, a1, a2)
    assert isclose(M2, 85.68, rtol=1e-3)
    # vector
    M2 = convert_polymer_standards([20.0, 100., 200.0], K1, K2, a1, a2)
    assert allclose(M2, [15.87,  85.68, 177.07], rtol=1e-3)


def test_reconstruct_Laguerre():
    d = Flory(100)
    moments = [d._moment_length(i) for i in range(4)]
    drec = reconstruct_Laguerre(moments)
    n = np.arange(1, 5*d.DPz, 1)
    pdf = drec(n)
    for i in range(len(moments)):
        assert isclose(np.dot(n**i, pdf), moments[i], rtol=1e-2)
