# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import pytest
from numpy import isclose, log10

from polykin.properties.vaporization import (
    DHVL_Kistiakowsky_Vetere,
    DHVL_Pitzer,
    DHVL_Vetere,
    DHVL_Watson,
    PL_Ambrose_Walton,
    PL_Lee_Kesler,
    PL_Wilson,
)


def test_hvap_pitzer():
    """Propionaldehyde. Example 7-6, page 220, Reid-Prausnitz-Poling."""
    hvap = DHVL_Pitzer(T=321.0, Tc=496.0, w=0.313)
    assert isclose(hvap, 28981, rtol=1e-4)


def test_hvapb_vetere():
    """Propionaldehyde. Example 7-8, page 227, Reid-Prausnitz-Poling."""
    hvap = DHVL_Vetere(Tb=321.0, Tc=496.0, Pc=47.6e5)
    assert isclose(hvap, 29144, rtol=1e-3)


def test_hvap_watson():
    """Hvap water."""
    hvap = DHVL_Watson(
        hvap1=2256,
        T1=373.0,
        T2=273.0,
        Tc=647.0,
    )
    assert isclose(hvap, 2500.0, rtol=5e-2)


def test_hvapb_kistiakowsky_vetere():
    """Several examples. Data from Reid-Prausnitz-Poling and NIST."""
    # Propionaldehyde. Example 7-9, page 231.
    hvap = DHVL_Kistiakowsky_Vetere(Tb=321.0, M=58.08e-3, kind="polar")
    assert isclose(hvap, 28710, rtol=1e-3)
    # ethanol
    hvap = DHVL_Kistiakowsky_Vetere(Tb=351.5, M=46.1e-3, kind="acid_alcohol")
    assert isclose(hvap, 42300, rtol=10e-2)
    # pentane
    hvap = DHVL_Kistiakowsky_Vetere(Tb=309.2, M=72.1e-3, kind="hydrocarbon")
    assert isclose(hvap, 26500, rtol=3e-2)
    # methyl acetate
    hvap = DHVL_Kistiakowsky_Vetere(Tb=330.0, M=74.1 - 3, kind="ester")
    assert isclose(hvap, 30300, rtol=10e-2)
    # pentane
    hvap = DHVL_Kistiakowsky_Vetere(Tb=309.2, M=72.1e-3, kind="any")
    assert isclose(hvap, 26500, rtol=10e-2)
    # exceptions
    with pytest.raises(ValueError):
        _ = DHVL_Kistiakowsky_Vetere(Tb=309.2, M=72.1e-3, kind="something")
    with pytest.raises(ValueError):
        _ = DHVL_Kistiakowsky_Vetere(Tb=309.2, kind="ester")


def test_PL_Lee_Kesler():
    """Example 7-1, page 208, Reid-Prausnitz-Poling."""
    Tb = 409.3
    Tc = 617.1
    Pc = 36.0e5
    for T, Pvap in zip([347.2, 460], [0.132e5, 3.353e5]):
        res = PL_Lee_Kesler(T, Tb, Tc, Pc)
        assert isclose(res, Pvap, rtol=2e-2)
    # trivial consistency check
    res = PL_Lee_Kesler(Tb, Tb, Tc, Pc)
    assert isclose(res, 101325.0)


def test_PL_Wilson():
    Tb = 418.3
    Tc = 647.0
    Pc = 39.9e5
    w = 0.257
    assert isclose(PL_Wilson(Tc, Tc, Pc, w), Pc)
    Pvap = PL_Wilson(Tc * 0.7, Tc, Pc, w)
    assert isclose(log10(Pc / Pvap) - 1.0, w, rtol=1e-3)
    assert isclose(PL_Wilson(Tb, Tc, Pc, w), 101325.0, rtol=5e-2)


def test_PL_Ambrose_Walton():
    """Acetone."""
    Tc = 508
    Pc = 47e5
    w = 0.309
    Pvap = PL_Ambrose_Walton(329.0, Tc, Pc, w)
    assert isclose(Pvap, 101325, rtol=5e-2)
