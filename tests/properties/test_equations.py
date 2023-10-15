# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.properties.dippr import \
    DIPPR100, DIPPR101, DIPPR102, DIPPR104, DIPPR105, DIPPR106, DIPPR107
from polykin.properties import Antoine, Wagner
from polykin import plotequations

import pytest
import numpy as np

# %% DIPPR equations


def test_DIPPR100():
    "CpL of water"
    p = DIPPR100(276370., -2090.1, 8.125, -0.014116, 0.0000093701)
    assert np.isclose(p(25., 'C')/18.02e3, 4.18, rtol=1e-3)


def test_DIPPR101():
    "P* of water"
    p = DIPPR101(73.649, -7258.2, -7.3037, 4.1653E-6, 2., unit='Pa')
    assert np.isclose(p(100., 'C'), 101325., rtol=1e-3)


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
    p = DIPPR105(0.14395, 0.0112, 649.727, 0.05107, unit='kg/m³')
    assert np.isclose(p(25., 'C'), 998., rtol=1e-3)


def test_DIPPR106():
    "DHvap of water"
    p = DIPPR106(647.096, 56600000., 0.612041, -0.625697, 0.398804)
    assert np.isclose(p(273.16, 'K'), 4.498084e7, rtol=1e-6)


def test_DIPPR107():
    "CpG of water"
    p = DIPPR107(33363., 26790., 2610.5, 8896., 1169., Tmin=100.,
                 Tmax=2273., unit='J/kmol.K')
    assert np.isclose(p(300., 'K'), 33585.904, rtol=1e-6)

# %% Vapor pressure equations


@pytest.fixture
def Pvap_water():
    """Pvap of water
    Parameters: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=4&Type=ANTOINE&Plot=on
    """
    return Antoine(A=4.6543, B=1435.264, C=-64.848, Tmin=255.9, Tmax=373.,
                   unit='bar')


def test_Antoine(Pvap_water):
    assert np.isclose(Pvap_water(100., 'C'), 1.01325, rtol=2e-2)


def test_Antoine_fit(Pvap_water):
    T = np.linspace(260, 373, 50)
    Y = Pvap_water(T)
    p = Antoine(A=1, B=1, C=1, unit='bar')
    result = p.fit(T, Y, logY=True)
    popt = result['parameters']
    assert np.isclose(popt['A'], 4.65, rtol=1e-2)
    assert np.isclose(popt['B'], 1435., rtol=1e-2)
    assert np.isclose(popt['C'], -64.85, rtol=1e-2)


def test_Antoine_fit_fitonly(Pvap_water):
    T = np.linspace(260, 373, 50)
    Y = Pvap_water(T)
    p = Antoine(A=1, B=1, C=0, unit='bar')
    result = p.fit(T, Y, fitonly=['A', 'B'], logY=True)
    popt = result['parameters']
    assert np.isclose(popt['A'], 6.207, rtol=1e-3)
    assert np.isclose(popt['B'], 2303., rtol=1e-3)


def test_Wagner():
    """Pvap of water
    Parameters: doi: 10.5541/ijot.372148
    """
    p = Wagner(Tc=647.096,
               Pc=220.64,  # bar
               a=-7.861942, b=1.879246, c=-2.266807,
               d=-2.128615,
               Tmin=273., Tmax=647., unit='bar')
    assert np.isclose(p(100., 'C'), 1.01325, rtol=2e-2)

# %% plot


def test_plotequations(Pvap_water):
    for kind in ['linear', 'semilogy', 'Arrhenius']:
        fig = plotequations([Pvap_water], kind=kind)
    assert fig is not None
