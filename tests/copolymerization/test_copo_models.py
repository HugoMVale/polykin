# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.copolymerization import TerminalModel, PenultimateModel
from polykin.kinetics import Arrhenius

import numpy as np

# %% Terminal


def test_CopoModel_repr():
    m = TerminalModel(1.2, 0.3)
    out = m.__repr__()
    assert out.startswith('name')


def test_TerminalModel_azeo():
    m = TerminalModel(1.2, 0.3)
    assert m.azeo is None
    m = TerminalModel(0.1, 0.1)
    assert m.azeo is not None
    assert np.isclose(m.azeo, 0.5)


def test_TerminalModel_F1():
    m = TerminalModel(0.1, 0.1)
    f1azeo = m.azeo
    assert f1azeo is not None
    assert np.isclose(m.F1(f1azeo), f1azeo)
    m = TerminalModel(1., 1.)
    assert np.all(np.isclose(m.F1([0.3, 0.7]), [0.3, 0.7]))
    m = TerminalModel(2., 0.6)
    f1 = [0.3, 0.7]
    assert np.all(m.F1(f1) > np.array(f1))


def test_TerminalModel_kp():
    k1 = Arrhenius(10, 0, 300, name='k1')
    k2 = Arrhenius(100, 0, 300, name='k2')
    m = TerminalModel(0.1, 0.1, k1, k2)
    T = 300.
    assert np.isclose(m.kp(1., T), k1(T))
    assert np.isclose(m.kp(0., T), k2(T))
    assert np.isclose(m.kp(0.5, T), k2(T))


def test_TerminalModel_drift():
    m = TerminalModel(0.1, 0.1)
    f1azeo = m.azeo
    assert f1azeo is not None
    f10 = [0.1, f1azeo, 0.9]
    f1_x = m.drift(f10, x=0.999)
    assert np.all(np.isclose(f1_x.flatten(), [0.0, f1azeo, 1]))


def test_TerminalModel_plot():
    model = TerminalModel(0.5, 0.5,
                          M1="Monomer1", M2="Monomer2", name="MyModel")
    result = model.plot('drift', M=1, f0=0.2, return_objects=True)
    assert result is not None and len(result) == 2
    result = model.plot('Mayo', M=1, return_objects=True)
    assert result is not None and len(result) == 2


# %% Penultimate

def test_PenultimateModel_azeo():
    m = PenultimateModel(2., 3., 0.5, 0.6, 1., 1.)
    assert m.azeo is None
    m = PenultimateModel(0.8, 0.8, 0.8, 0.8, 1., 1.)
    assert m.azeo is not None
    assert np.isclose(m.azeo, 0.5)


def test_PenultimateModel_F1():
    m = PenultimateModel(0.4, 0.4, 0.8, 0.8, 1., 1.)
    f1azeo = m.azeo
    assert f1azeo is not None
    assert np.isclose(m.F1(f1azeo), f1azeo)
    m = PenultimateModel(1, 1, 1, 1, 1, 1)
    f1 = [0.3, 0.7]
    assert np.all(np.isclose(m.F1(f1), f1))
    m = PenultimateModel(2.3, 2.0, 0.5, 0.6, 1, 1)
    assert np.all(m.F1(f1) > np.array(f1))
    m = PenultimateModel(0.5, 0.6, 2.3, 2.0, 1, 1)
    assert np.all(m.F1(f1) < np.array(f1))


def test_PenultimateModel_kp():
    k1 = Arrhenius(10, 0, 300, name='k1')
    k2 = Arrhenius(100, 0, 300, name='k2')
    m = PenultimateModel(0.1, 0.1, 0.1, 0.1, 1., 1., k1, k2)
    T = 300.
    assert np.isclose(m.kp(1., T), k1(T))
    assert np.isclose(m.kp(0., T), k2(T))
    assert np.isclose(m.kp(0.5, T), k2(T))
