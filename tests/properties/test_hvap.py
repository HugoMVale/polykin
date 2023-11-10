# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.properties.vaporization_enthalpy import \
    DHVL_pitzer, DHVL_vetere, DHVL_watson, DHVL_kistiakowsky_vetere

import numpy as np


def test_hvap_pitzer():
    "Propionaldehyde. Example 7-6, page 220."
    hvap = DHVL_pitzer(T=321., Tc=496., w=0.313)
    assert np.isclose(hvap, 28981, rtol=1e-4)


def test_hvapb_vetere():
    "Propionaldehyde. Example 7-8, page 227."
    hvap = DHVL_vetere(Tb=321., Tc=496., Pc=47.6e5)
    assert np.isclose(hvap, 29144, rtol=1e-3)


def test_hvap_watson():
    "Hvap water."
    hvap = DHVL_watson(hvap1=2256, T1=373., T2=273., Tc=647.,)
    assert np.isclose(hvap, 2500., rtol=5e-2)


def test_hvapb_kistiakowsky_vetere():
    "Several examples. data from Reid-Prausnitz-Poling and NIST."
    # Propionaldehyde. Example 7-9, page 231.
    hvap = DHVL_kistiakowsky_vetere(Tb=321., M=58.08e-3, kind='polar')
    assert np.isclose(hvap, 28710, rtol=1e-3)
    # ethanol
    hvap = DHVL_kistiakowsky_vetere(Tb=351.5, M=46.1e-3, kind='acid_alcohol')
    assert np.isclose(hvap, 42300, rtol=10e-2)
    # pentane
    hvap = DHVL_kistiakowsky_vetere(Tb=309.2, M=72.1e-3, kind='hydrocarbon')
    assert np.isclose(hvap, 26500, rtol=3e-2)
    # methyl acetate
    hvap = DHVL_kistiakowsky_vetere(Tb=330.0, M=74.1-3, kind='ester')
    assert np.isclose(hvap, 30300, rtol=10e-2)
    # pentane
    hvap = DHVL_kistiakowsky_vetere(Tb=309.2, M=72.1e-3, kind='any')
    assert np.isclose(hvap, 26500, rtol=10e-2)
