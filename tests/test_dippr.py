from polykin.physprops import \
    DIPPR100, DIPPR101, DIPPR102, DIPPR104, DIPPR105, DIPPR106


# import pytest
import numpy as np


def test_DIPPR100():
    "CpL of water"
    p = DIPPR100(276370., -2090.1, 8.125, -0.014116, 0.0000093701)
    assert np.isclose(p(25.)/18.02e3, 4.18, rtol=1e-3)


def test_DIPPR101():
    "P* of water"
    p = DIPPR101(73.649, -7258.2, -7.3037, 4.1653E-6, 2.)
    assert np.isclose(p(100.), 101325., rtol=1e-3)


def test_DIPPR102():
    "Viscosity of pentane vapor"
    p = DIPPR102(6.3412e-08, 0.84758, 41.718)
    assert np.isclose(p(1000., 'K'), 2.12403e-5, rtol=1e-6)


def test_DIPPR104():
    "2nd virial coefficient of water vapor"
    p = DIPPR104(0.02222, -26.38, -16750000, -3.894E19, 3.133E21)
    assert np.isclose(p(273.15, 'K'), -1.782685, rtol=1e-6)


def test_DIPPR105():
    "rhoL of water"
    p = DIPPR105(0.14395, 0.0112, 649.727, 0.05107)
    assert np.isclose(p(25.), 998., rtol=1e-3)


def test_DIPPR106():
    "DHvap of water"
    p = DIPPR106(647.096, 56600000., 0.612041, -0.625697, 0.398804)
    assert np.isclose(p(273.16, 'K'), 4.498084e7, rtol=1e-6)
