# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from numpy import allclose, isclose

from polykin.distributions import convert_polymer_standards


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
