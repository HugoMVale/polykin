# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Any

import numpy as np
import pytest
from numpy import all, isclose

from polykin.thermo.eos import (IdealGas, PengRobinson, RedlichKwong, Soave,
                                Virial)
from polykin.utils.exceptions import ShapeError


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


def test_IdealGas(ideal_gas: IdealGas, air_parameters: dict[str, Any]):
    eos = ideal_gas
    state = air_parameters['state']
    assert isclose(eos.Z(**state), 1.)
    assert isclose(eos.v(**state), 24.4e-3, rtol=1e-2)
    assert isclose(eos.P(state['T'], eos.v(
        **state), state['y']), state['P'], rtol=1e-2)


def test_air(air_parameters: dict[str, Any], ideal_gas: IdealGas):
    p = air_parameters['p']
    state = air_parameters['state']
    for EOS in [Virial]:
        eos = EOS(**p)
        assert isclose(ideal_gas.Z(**state), eos.Z(**state), rtol=1e-2)
        assert isclose(ideal_gas.v(**state), eos.v(**state), rtol=1e-2)
        assert isclose(eos.P(state['T'], eos.v(
            **state), state['y']), state['P'], rtol=1e-2)  # type: ignore


def test_Virial_validation():
    with pytest.raises(ShapeError):
        _ = Virial(Tc=200., Pc=50e5, Zc=0.28, w=[0.2, 0.3])


def test_Virial_Z():
    "Example 3-1, p. 35, Reid-Prausnitz-Poling."
    eos = Virial(Tc=385.0, Pc=41.4e5, Zc=0.28, w=0.204)
    state = (366.5, 20.67e5, np.array([1.]))
    assert isclose(eos.Z(*state), 0.75, rtol=0.1)
    assert isclose(eos.v(*state), 1097e-6, rtol=0.1)


def test_Virial_B():
    "Example 3-2, p. 41, Reid-Prausnitz-Poling."
    eos = Virial(Tc=571., Pc=32.7e5, Zc=0.28, w=0.385)
    assert isclose(eos.Bm(273.15+120., np.array([1.])), -1580e-6, rtol=0.2)


def test_Virial_Bij():
    "Example 4-1, p. 81, Reid-Prausnitz-Poling."
    eos = Virial(Tc=[190.6, 425.2], Pc=[46.0e5, 38.e5],
                 Zc=[0.288, 0.274], w=[0.012, 0.199])
    T = 443.
    y = np.array([0.5, 0.5])
    assert isclose(eos.Bm(T, y), -107e-6, rtol=0.05)
    Bij = eos.Bij(T)
    assert isclose(Bij[0, 0], -8.1e-6, rtol=0.1)
    assert isclose(Bij[1, 1], -293.4e-6, rtol=0.1)
    assert isclose(Bij[0, 1], -58.6e-6, rtol=0.1)


def test_Virial_phi():
    "Example 10.8, p. 347, Smith-Van Ness-Abbott."
    # a mix of list, tuple and array is used on purpose
    eos = Virial(Tc=[535.5, 591.8], Pc=(41.5e5, 41.1e5), Zc=[0.249, 0.264],
                 w=np.array([0.323, 0.262]))
    y = np.array([0.5, 0.5])
    T = 273.15 + 50.
    P = 25e3
    assert all(isclose(eos.phiV(T, P, y), [0.987, 0.983], rtol=1e-3))
    assert all(isclose(eos.fV(T, P, y), np.array(
        [0.987, 0.983])*P/2, rtol=1e-3))


def test_Virial_isopropanol():
    "Example 3.6, p. 78, Smith-Van Ness-Abbott."
    eos = Virial(Tc=508.3, Pc=47.6e5, Zc=0.248, w=0.665)
    T = 273.15 + 200.
    P = 10e5
    y = np.array([1.])
    assert isclose(eos.Bm(T, y), -388e-6, rtol=0.05)
    assert isclose(eos.Z(T, P, y), 0.9014, rtol=0.05)
    assert isclose(eos.v(T, P, y), 3.539e-3, rtol=0.05)


def test_Virial_butene():
    """Example 6.6, p. 210, Smith-Van Ness-Abbott.
    Example 10.7, p. 343, Smith-Van Ness-Abbott."""
    eos = Virial(Tc=420.0, Pc=(40.43e5,), Zc=0.28, w=[0.191])
    y = np.array([1.])
    assert isclose(eos.Z(273.15 + 200., 70e5, y), 0.512, rtol=0.2)
    T = 273.15
    P = 1.2771e5
    assert isclose(eos.phiV(T, P, y), 0.956, rtol=0.001)
    DX = eos.DX(T, P, y, P0=P)
    assert isclose(DX['S'], -0.8822, rtol=0.01)
    assert isclose(DX['H'], -344, rtol=0.01)


def test_RK():
    "Example 3.7, p. 84, Smith-Van Ness-Abbott."
    eos = RedlichKwong(Tc=[416.3], Pc=[66.8e5])
    T = 273.15 + 60.
    P = 13.76e5
    y = np.array([1.])
    assert isclose(eos.b, 44.891e-6, rtol=1e-3)
    assert all(isclose(eos.v(T, P, y), [71.34e-6, 1713e-6], rtol=1e-3))


def test_Virial_RK_ammonia():
    "Example 3.10, p. 93, Smith-Van Ness-Abbott."
    ideal = IdealGas()
    rk = RedlichKwong(Tc=[405.7], Pc=[112.8e5])
    virial = Virial(Tc=[405.7], Pc=[112.8e5], Zc=[0.28], w=[.253])
    T = 273.15 + 65.
    v = 1021.2e-6
    y = np.array([1.])
    assert isclose(ideal.P(T, v), 27.53e5, rtol=1e-3)
    assert isclose(rk.P(T, v, y), 23.84e5, rtol=1e-3)
    assert isclose(rk.b, 25.91e-6, rtol=1e-3)
    assert isclose(virial.P(T, v, y), 23.76e5, rtol=1e-3)


def test_Cubic_Z():
    "Example 3-3, p. 46, Reid-Prausnitz-Poling."
    Tc = [408.2]
    Pc = [36.5e5]
    w = [0.183]
    state = {'T': 300., 'y': np.array([1.])}
    eos = Soave(Tc, Pc, w)
    Z = eos.Z(**state, P=3.706e5)
    assert all(isclose(Z, (0.01687, 0.9057), rtol=1e-3))
    eos = PengRobinson(Tc, Pc, w)
    Z = eos.Z(**state, P=3.683e5)
    assert all(isclose(Z, (0.01479, 0.9015), rtol=1e-3))
    eos = RedlichKwong(Tc, Pc)
    Z = eos.Z(**state, P=3.706e5)
    assert all(isclose(Z, (0.01687, 0.9057), rtol=0.05))


def test_Cubic_butene():
    "Example 10.7, p. 343, Smith-Van Ness-Abbott."
    y = np.array([1.])
    T = 273.15 + 200.
    P = 70e5
    for EOS in [Soave, PengRobinson]:
        eos = EOS(Tc=[420.0], Pc=[40.43e5], w=[0.191])
        assert isclose(eos.Z(T, P, y), 0.512, rtol=0.1)
        assert isclose(eos.phiV(T, P, y), 0.638, rtol=0.05)
        assert isclose(eos.fV(T, P, y), 44.7e5, rtol=0.05)


def test_Cubic_phi():
    "Example 10.8, p. 347, Smith-Van Ness-Abbott."
    Tc = [535.5, 591.8]
    Pc = [41.5e5, 41.1e5]
    w = [0.323, 0.262]
    rk = RedlichKwong(Tc, Pc)
    srk = Soave(Tc, Pc, w)
    pr = PengRobinson(Tc, Pc, w)
    y = np.array([0.5, 0.5])
    T = 273.15 + 50
    P = 25e3
    for eos in [rk, srk, pr]:
        assert all(isclose(eos.phiV(T, P, y), [0.987, 0.983], rtol=1e-2))


def test_Cubic_B():
    "Isopropanol"
    Tc = [508.3]
    Pc = [47.6e5]
    Zc = [0.248]
    w = [0.665]
    virial = Virial(Tc, Pc, Zc, w)
    srk = Soave(Tc, Pc, w)
    T = 273.15 + 200.
    y = np.array([1.])
    assert isclose(virial.Bm(T, y), srk.Bm(T, y), rtol=0.1)


def test_Z_Cubic_interaction():
    Tc = [282.4, 126.2]
    Pc = [50.4e5, 33.9e5]
    w = [0.089, 0.039]
    k = np.array([[0., 0.080], [0., 0.]])  # Reid, p. 83
    T = 350.
    P = 100e5
    y = np.array([0.5, 0.5])
    eos1 = PengRobinson(Tc, Pc, w, k=None)
    eos2 = PengRobinson(Tc, Pc, w, k=k)
    assert isclose(eos1.Z(T, P, y), eos2.Z(T, P, y), rtol=0.01)


def test_Cubic_departures():
    "Example 5-3, p. 111, Reid-Prausnitz-Poling."
    # isobutane
    Tc = [364.9]
    Pc = [46.0e5]
    w = [0.144]
    M = 42.081
    T = 398.15
    P = 100e5
    y = np.array([1.])
    for EOS in [Soave, PengRobinson]:
        eos = EOS(Tc, Pc, w)
        DX = eos.DX(T, P, y)
        assert isclose(DX['H']/M, -235, rtol=0.01)
        assert isclose(DX['S']/M, -1.37, rtol=0.01)
