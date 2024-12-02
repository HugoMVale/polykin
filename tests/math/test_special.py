# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import inf, isclose, pi

from polykin.math.special import i2erfc, ierfc, roots_xtanx
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


def test_roots_xtanx():
    x = roots_xtanx(0, 4)
    assert all(isclose(x, [pi*i for i in range(len(x))]))
    x = roots_xtanx(0.01, 6)
    assert all(isclose(x, [0.0998, 3.1448, 6.2848, 9.4258, 12.5672, 15.7086],
                       atol=1e-4))
    x = roots_xtanx(1, 6)
    assert all(isclose(x, [0.8603, 3.4256, 6.4373, 9.5293, 12.6453, 15.7713],
                       atol=1e-4))
    x = roots_xtanx(1e2, 6)
    assert all(isclose(x, [1.5552, 4.6658, 7.7764, 10.8871, 13.9981, 17.1093],
                       atol=1e-4))
    x = roots_xtanx(1e5, 6)
    assert all(isclose(x, [1.5708, 4.7124, 7.8540, 10.9956, 14.1372, 17.2788],
                       atol=1e-4))
