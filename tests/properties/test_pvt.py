# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.properties.eos import IdealGas, Virial

import pytest
import numpy as np
# from scipy.constants import R


@pytest.fixture
def ideal_gas():
    return IdealGas()


@pytest.fixture
def air_parameters():
    Tc = [126.2, 154.6]
    Pc = [33.9e5, 50.4e5]
    Zc = [0.290, 0.288]
    w = [0.039, 0.025]
    y = np.array([0.79, 0.21])
    T = 298.15
    P = 1.01325e5
    return {'p': {'Tc': Tc, 'Pc': Pc, 'Zc': Zc, 'w': w},
            'state': {'T': T, 'P': P, 'y': y}}


def test_IdealGas(ideal_gas, air_parameters):
    eos = ideal_gas
    state = air_parameters['state']
    assert np.isclose(eos.Z(**state), 1.)
    assert np.isclose(eos.V(**state), 24.4e-3, rtol=1e-2)
    assert np.isclose(eos.P(state['T'], eos.V(
        **state), state['y']), state['P'], rtol=1e-2)


def test_air(air_parameters, ideal_gas):
    p = air_parameters['p']
    state = air_parameters['state']
    for EOS in [Virial]:
        eos = EOS(**p)
        assert np.isclose(ideal_gas.Z(**state), eos.Z(**state), rtol=1e-2)
        assert np.isclose(ideal_gas.V(**state), eos.V(**state), rtol=1e-2)
        assert np.isclose(eos.P(state['T'], eos.V(
            **state), state['y']), state['P'], rtol=1e-2)  # type: ignore


def test_Virial2_Z():
    "Example 3-1, p. 35, Reid-Prausnitz-Poling."
    eos = Virial(Tc=[385.0], Pc=[41.4e5], Zc=[0.28], w=[0.204])
    state = (366.5, 20.67e5, np.array([1.]))
    assert np.isclose(eos.Z(*state), 0.75, rtol=0.1)
    assert np.isclose(eos.V(*state), 1097e-6, rtol=0.1)


def test_Virial2_B():
    "Example 3-2, p. 41, Reid-Prausnitz-Poling."
    eos = Virial(Tc=[571.], Pc=[32.7e5], Zc=[0.28], w=[0.385])
    assert np.isclose(eos.Bm(273.15+120., np.array([1.])), -1580e-6, rtol=0.2)


def test_Virial2_Bij():
    "Example 4-1, p. 81, Reid-Prausnitz-Poling."
    eos = Virial(Tc=[190.6, 425.2], Pc=[46.0e5, 38.e5],
                 Zc=[0.288, 0.274], w=[0.012, 0.199])
    T = 443.
    y = np.array([0.5, 0.5])
    assert np.isclose(eos.Bm(T, y), -107e-6, rtol=0.05)
    Bij = eos.Bij(T)
    assert np.isclose(Bij[0, 0], -8.1e-6, rtol=0.1)
    assert np.isclose(Bij[1, 1], -293.4e-6, rtol=0.1)
    assert np.isclose(Bij[0, 1], -58.6e-6, rtol=0.1)

# def test_Z_cubic():
#     "Example 3-3, p. 46"
#     y = np.array([1.])
#     Tc = np.array([408.2])
#     Pc = np.array([36.5e5])
#     w = np.array([0.183])
#     kwargs = {'T': 300., 'y': y, 'Tc': Tc, 'Pc': Pc, 'w': w}
#     Z = Z_cubic(**kwargs, P=3.706e5, method='SRK')
#     assert np.all(np.isclose(Z, (0.01687, 0.9057), rtol=0.01))
#     Z = Z_cubic(**kwargs, P=3.683e5, method='PR')
#     assert np.all(np.isclose(Z, (0.01479, 0.9015), rtol=0.01))
#     Z = Z_cubic(**kwargs, P=3.706e5, method='RK')
#     assert np.all(np.isclose(Z, (0.01687, 0.9057), rtol=0.05))


# def test_Z_cubic_interaction():
#     y = np.array([0.5, 0.5])         # mol/mol
#     Tc = np.array([282.4, 126.2])    # K
#     Pc = np.array([50.4e5, 33.9e5])  # Pa
#     w = np.array([0.089, 0.039])
#     k = np.array([[0., 0.080], [0., 0.]])  # Reid, p. 83
#     T = 350.
#     P = 100e5
#     kwargs = {'T': T, 'P': P, 'y': y, 'Tc': Tc, 'Pc': Pc, 'w': w}
#     Z1 = Z_cubic(**kwargs, k=k, method='PR')
#     Z2 = Z_cubic(**kwargs, k=None, method='PR')
#     assert np.isclose(Z1, Z2, rtol=0.01)
#     V = Z1*R*T/P
#     assert (np.isclose(V, 2.61e-4, rtol=0.01))
