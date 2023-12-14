# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.properties.pvt import Z_virial, Z_cubic

# import pytest
import numpy as np
from scipy.constants import R


def test_Z_virial():
    "Example 3-1, p. 35"
    Z = Z_virial(T=366.5, P=20.67, Tc=385.0, Pc=41.4, w=0.204)
    assert np.isclose(Z, 0.75, rtol=0.1)


def test_Z_cubic():
    "Example 3-3, p. 46"
    y = np.array([1.])
    Tc = np.array([408.2])
    Pc = np.array([36.5e5])
    w = np.array([0.183])
    kwargs = {'T': 300., 'y': y, 'Tc': Tc, 'Pc': Pc, 'w': w}
    Z = Z_cubic(**kwargs, P=3.706e5, method='SRK')
    assert np.all(np.isclose(Z, (0.01687, 0.9057), rtol=0.01))
    Z = Z_cubic(**kwargs, P=3.683e5, method='PR')
    assert np.all(np.isclose(Z, (0.01479, 0.9015), rtol=0.01))
    Z = Z_cubic(**kwargs, P=3.706e5, method='RK')
    assert np.all(np.isclose(Z, (0.01687, 0.9057), rtol=0.05))


def test_Z_cubic_interaction():
    y = np.array([0.5, 0.5])         # mol/mol
    Tc = np.array([282.4, 126.2])    # K
    Pc = np.array([50.4e5, 33.9e5])  # Pa
    w = np.array([0.089, 0.039])
    k = np.array([[0., 0.080], [0., 0.]])  # Reid, p. 83
    T = 350.
    P = 100e5
    kwargs = {'T': T, 'P': P, 'y': y, 'Tc': Tc, 'Pc': Pc, 'w': w}
    Z1 = Z_cubic(**kwargs, k=k, method='PR')
    Z2 = Z_cubic(**kwargs, k=None, method='PR')
    assert np.isclose(Z1[0], Z2[0], rtol=0.01)
    assert Z1[-1] is np.nan
    V = Z1[0]*R*T/P
    assert (np.isclose(V, 2.61e-4, rtol=0.01))
