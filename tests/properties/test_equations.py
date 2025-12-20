# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
import pytest
from numpy import isclose

from polykin.properties import plotequations
from polykin.properties.equations.dippr import (
    DIPPR100,
    DIPPR101,
    DIPPR102,
    DIPPR104,
    DIPPR105,
    DIPPR106,
    DIPPR107,
)
from polykin.properties.equations.vapor_pressure import Antoine, Wagner25, Wagner36
from polykin.properties.equations.viscosity import Yaws

# %% DIPPR equations


def test_DIPPR100():
    """CpL of water."""
    p = DIPPR100(276370.0, -2090.1, 8.125, -0.014116, 0.0000093701)
    assert isclose(p(25.0, "C") / 18.02e3, 4.18, rtol=1e-3)


def test_DIPPR101():
    """P* of water."""
    p = DIPPR101(73.649, -7258.2, -7.3037, 4.1653e-6, 2.0, unit="Pa")
    assert isclose(p(100.0, "C"), 101325.0, rtol=1e-3)


def test_DIPPR102():
    """Viscosity of pentane vapor."""
    p = DIPPR102(6.3412e-08, 0.84758, 41.718)
    assert isclose(p(1000.0, "K"), 2.12403e-5, rtol=1e-6)


def test_DIPPR104():
    """2nd virial coefficient of water vapor."""
    p = DIPPR104(0.02222, -26.38, -16750000, -3.894e19, 3.133e21)
    assert isclose(p(273.15, "K"), -1.782685, rtol=1e-6)


def test_DIPPR105():
    """RhoL of water."""
    p = DIPPR105(0.14395, 0.0112, 649.727, 0.05107, unit="kg/m³")
    assert isclose(p(25.0, "C"), 998.0, rtol=1e-3)


def test_DIPPR106():
    """DHvap of water."""
    p = DIPPR106(647.096, 56600000.0, 0.612041, -0.625697, 0.398804)
    assert isclose(p(273.16, "K"), 4.498084e7, rtol=1e-6)


def test_DIPPR107():
    """CpG of water."""
    p = DIPPR107(
        33363.0,
        26790.0,
        2610.5,
        8896.0,
        1169.0,
        Tmin=100.0,
        Tmax=2273.0,
        unit="J/kmol.K",
    )
    assert isclose(p(300.0, "K"), 33585.904, rtol=1e-6)


# %% Vapor pressure equations


@pytest.fixture
def Pvap_water():
    """Pvap of water.

    Parameters: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=4&Type=ANTOINE&Plot=on
    """
    return Antoine(A=4.6543, B=1435.264, C=-64.848, Tmin=255.9, Tmax=373.0, unit="bar")


def test_PropertyEquationT_repr(Pvap_water):
    out = Pvap_water.__repr__()
    assert out.startswith("name")


def test_Antoine(Pvap_water):
    assert isclose(Pvap_water(100.0, "C"), 1.01325, rtol=2e-2)


def test_Antoine_fit(Pvap_water):
    T = np.linspace(260.0, 373.0, 50)
    Y = Pvap_water(T)
    p = Antoine(A=1, B=1, C=1, unit="bar")
    result = p.fit(T, Y, logY=True)
    popt = result["parameters"]
    assert isclose(popt["A"], 4.65, rtol=1e-2)
    assert isclose(popt["B"], 1435.0, rtol=1e-2)
    assert isclose(popt["C"], -64.85, rtol=1e-2)


def test_Antoine_fit_fitonly(Pvap_water):
    T = np.linspace(260.0, 373.0, 50).tolist()
    Y = Pvap_water(T).tolist()
    p = Antoine(A=1, B=1, C=0, unit="bar")
    result = p.fit(T, Y, fit_only=["A", "B"], logY=True)
    popt = result["parameters"]
    assert isclose(popt["A"], 6.207, rtol=1e-3)
    assert isclose(popt["B"], 2303.0, rtol=1e-3)


def test_Wagner25():
    """Pvap of water.

    Parameters: doi: 10.5541/ijot.372148
    """
    p = Wagner25(
        Tc=647.096,
        Pc=220.64,  # bar
        a=-7.861942,
        b=1.879246,
        c=-2.266807,
        d=-2.128615,
        Tmin=273.0,
        Tmax=647.0,
        unit="bar",
    )
    assert isclose(p(100.0, "C"), 1.01325, rtol=1e-2)


def test_Wagner36():
    """Pvap of water.

    Parameters: Reid-Prausnitz, p. 669.
    """
    p = Wagner36(
        Tc=647.096,
        Pc=220.64,  # bar
        a=-7.76451,
        b=1.45838,
        c=-2.77580,
        d=-1.23303,
        Tmin=273.0,
        Tmax=647.0,
        unit="bar",
    )
    assert isclose(p(100.0, "C"), 1.01325, rtol=1e-2)


# %% Viscosity


def test_Yaws():
    """Viscosity of styrene."""
    mu = Yaws(
        A=-2.717,
        B=9.461e2,
        C=-3.173e-3,
        D=1.683e-6,
        base10=False,
        Tmin=243.0,
        Tmax=633.0,
        unit="cP",
    )
    assert isclose(mu(25, "C"), 0.71, rtol=2e-2)


def test_Antoine_Yaws_():
    """Input validation."""
    p = (123.0, 678.0)
    T = 300.0
    for Equation in [Antoine, Yaws]:
        y1 = Equation(*p, base10=False)
        y2 = Equation(*p, base10=True)
        assert isclose(np.log(y1(T)), np.log10(y2(T)), rtol=2e-2)


# %% Plot


def test_plotequations(Pvap_water):
    for kind in ["linear", "semilogy", "Arrhenius"]:
        fig = plotequations([Pvap_water], kind=kind)
    assert fig is not None
