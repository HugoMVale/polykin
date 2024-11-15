# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import inf, isclose

from polykin.math.special import i2erfc, ierfc
from polykin.utils.math import huge


def test_ierfc():
    "Crank, p. 375"
    assert isclose(2*ierfc(0), 1.1284, atol=1e-4)
    assert isclose(2*ierfc(0.5), 0.3993, atol=1e-4)
    assert isclose(ierfc(10), 0)
    assert isclose(ierfc(huge), 0)
    assert isclose(ierfc(inf), 0)


def test_i2erfc():
    "Crank, p. 376"
    assert isclose(4*i2erfc(0.5), 0.2799, atol=1e-4)
    assert isclose(i2erfc(10), 0)
    assert isclose(i2erfc(huge), 0)
    assert isclose(i2erfc(inf), 0)
