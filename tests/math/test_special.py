# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import isclose

from polykin.math.special import ierfc, i2erfc


def test_ierfc():
    "Crank, p. 375"
    assert isclose(2*ierfc(0.5), 0.3993, atol=1e-4)


def test_i2erfc():
    "Crank, p. 376"
    assert isclose(4*i2erfc(0.5), 0.2799, atol=1e-4)
