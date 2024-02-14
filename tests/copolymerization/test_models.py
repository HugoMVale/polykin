# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from numpy import all, isclose

from polykin.copolymerization import (ImplicitPenultimateModel,
                                      PenultimateModel, TerminalModel)
from polykin.kinetics import Arrhenius

# %% Terminal


def test_CopoModel_input_validation(capsys):
    _ = TerminalModel(1.2, 1.3)
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))


def test_CopoModel_repr():
    m = TerminalModel(1.2, 0.3)
    out = m.__repr__()
    assert out.startswith('name')


def test_TerminalModel_azeo():
    m = TerminalModel(1.2, 0.3)
    assert m.azeotrope is None
    m = TerminalModel(0.1, 0.1)
    assert m.azeotrope is not None
    assert isclose(m.azeotrope, 0.5)


def test_TerminalModel_F1():
    m = TerminalModel(0.1, 0.1)
    f1azeo = m.azeotrope
    assert f1azeo is not None
    assert isclose(m.F1(f1azeo), f1azeo)
    m = TerminalModel(1., 1.)
    assert all(isclose(m.F1([0.3, 0.7]), [0.3, 0.7]))
    m = TerminalModel(2., 0.6)
    f1 = [0.3, 0.7]
    assert all(m.F1(f1) > np.array(f1))


def test_TerminalModel_kp():
    k1 = Arrhenius(10, 0, 300, name='k1')
    k2 = Arrhenius(100, 0, 300, name='k2')
    m = TerminalModel(0.1, 0.1, k1, k2)
    T = 300.
    assert isclose(m.kp(1., T), k1(T))
    assert isclose(m.kp(0., T), k2(T))
    assert isclose(m.kp(0.5, T), k2(T))


def test_TerminalModel_triads():
    "Example 4.1, p. 179, Dotson-Galván-Laurence-Tirell"
    m = TerminalModel(r1=0.48, r2=0.42, M1='ST', M2='MMA')
    f1 = 0.75
    F1 = m.F1(f1)
    triads = np.array(list(m.triads(f1=0.75).values()))
    triads1 = triads[0:3]
    triads2 = triads[3:]
    assert all(isclose(triads1/F1, [0.348, 0.484, 0.168], rtol=1e-2))
    assert all(isclose(triads2/(1. - F1), [0.0151, 0.215, 0.769], rtol=1e-2))


def test_TerminalModel_sld():
    "Figure 4.3, 4.4 p. 184, Dotson-Galván-Laurence-Tirell"
    tm = TerminalModel(r1=0.331, r2=0.053, M1='ST', M2='ACN')
    pm = PenultimateModel(r11=0.229, r12=0.091, r21=0.634, r22=0.039,
                          s1=1, s2=1, M1='ST', M2='ACN')
    f1 = 0.6
    k = np.arange(1, 100)
    for model in [tm, pm]:
        F1 = model.F1(f1)
        Sn = model.sequence(f1, k=None)
        assert isclose(F1/(1 - F1), Sn['1']/Sn['2'])
        Sk = model.sequence(f1, k=k)
        assert all(isclose(list(Sn.values()),
                           [np.dot(k, Sk['1']), np.dot(k, Sk['2'])],
                           rtol=1e-5))
    Sk = tm.sequence(f1, k=1)
    assert all(isclose(list(Sk.values()), [0.668, 0.966], rtol=1e-2))


def test_TerminalModel_drift():
    m = TerminalModel(0.1, 0.1)
    f1azeo = m.azeotrope
    assert f1azeo is not None
    f10 = [0.1, f1azeo, 0.9]
    f1_x = m.drift(f10, x=0.999)
    assert all(isclose(f1_x.flatten(), [0.0, f1azeo, 1]))


def test_TerminalModel_plot():
    m = TerminalModel(0.5, 0.5, M1="Monomer1", M2="Monomer2", name="MyModel")
    result = m.plot('drift', M=1, f0=0.2, return_objects=True)
    assert result is not None and len(result) == 2
    result = m.plot('Mayo', M=1, return_objects=True)
    assert result is not None and len(result) == 2


def test_TerminalModel_from_Qe():
    m = TerminalModel.from_Qe((1., -0.8), (0.78, 0.4), M1='STY', M2='MMA')
    assert m.azeotrope and isclose(m.azeotrope, 0.5, atol=0.1)
    assert all(isclose([m.r1, m.r2], [0.5, 0.5], atol=0.1))

# %% Penultimate


def test_PenultimateModel_azeo():
    m = PenultimateModel(2., 3., 0.5, 0.6, 1., 1.)
    assert m.azeotrope is None
    m = PenultimateModel(0.8, 0.8, 0.8, 0.8, 1., 1.)
    assert m.azeotrope is not None
    assert isclose(m.azeotrope, 0.5)


def test_PenultimateModel_F1():
    m = PenultimateModel(0.4, 0.4, 0.8, 0.8, 1., 1.)
    f1azeo = m.azeotrope
    assert f1azeo is not None
    assert isclose(m.F1(f1azeo), f1azeo)
    m = PenultimateModel(1, 1, 1, 1, 1, 1)
    f1 = [0.3, 0.7]
    assert all(isclose(m.F1(f1), f1))
    m = PenultimateModel(2.3, 2.0, 0.5, 0.6, 1, 1)
    assert all(m.F1(f1) > np.array(f1))
    m = PenultimateModel(0.5, 0.6, 2.3, 2.0, 1, 1)
    assert all(m.F1(f1) < np.array(f1))


def test_PenultimateModel_kp():
    k1 = Arrhenius(10, 0, 300, name='k1')
    k2 = Arrhenius(100, 0, 300, name='k2')
    m = PenultimateModel(0.1, 0.1, 0.1, 0.1, 1., 1., k1, k2)
    T = 300.
    assert isclose(m.kp(1., T), k1(T))
    assert isclose(m.kp(0., T), k2(T))
    assert isclose(m.kp(0.5, T), k2(T))


# %% Implicit penultimate

def test_PenultimateModel_ImplicitPenultimateModel_kp():
    k1 = Arrhenius(10, 1000, 300, name='k1')
    k2 = Arrhenius(100, 2000, 300, name='k2')
    r1 = 0.4
    r2 = 0.8
    s1 = 0.6
    s2 = 1.5
    im = ImplicitPenultimateModel(r1, r2, s1, s2, k1, k2)
    pm = PenultimateModel(r1, r2, r1, r2, s1, s2, k1, k2)
    for f10 in [0.2, 0.45, 0.8]:
        for T in [280., 300., 350.]:
            assert isclose(im.F1(f10), pm.F1(f10))
            assert isclose(im.kp(f10, T), pm.kp(f10, T))
