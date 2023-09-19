from polykin.coefficients import Arrhenius, Eyring, \
    TerminationCompositeModel, PropagationHalfLength,\
    DIPPR100, DIPPR101, DIPPR102, DIPPR104, DIPPR105, DIPPR106
from polykin.utils import RangeError

import pytest
import numpy as np

# %% Arrhenius and Eyring


def test_input_validation(capsys):
    with pytest.raises(ValueError):
        _ = Arrhenius(-1, 1, 1)
    with pytest.raises(ValueError):
        _ = Arrhenius(1, 1, -1)
    with pytest.raises(ValueError):
        _ = Arrhenius(1, 1, 1, -1)
    with pytest.raises(ValueError):
        _ = Arrhenius(1, 1, 1, 1, -1)
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [1, 2, 3], 298)
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [2, 3], [200])
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [2, 3], Tmin=[200])
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [2, 3], Tmax=[500])
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [2, 3], [500, -100])
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [2, 3], [300, 300], [200, 300], [150, 500])
    with pytest.raises(ValueError):
        _ = Eyring([-1e3, 2e3], [1e3, 2e3], [0.5, 1])
    with pytest.raises(ValueError):
        _ = Eyring([1e3, 2e3], [-1e3, 2e3], [0.5, 1])
    with pytest.raises(ValueError):
        _ = Eyring([1e3, 2e4], [1e3, 2e3], [2, .5])
    with pytest.raises(ValueError):
        _ = Eyring([1e3, 2e4], [1e3, 2e3], [0.4, .5, 0.8])
    with pytest.raises(ValueError):
        _ = Eyring([1e3, 2e4], [1e3, 2e3], [0.4, 0.8], Tmin=[-1, 100])
    with pytest.raises(ValueError):
        _ = Eyring([1e3, 2e4], [1e3, 2e3], [0.4, 0.8], Tmax=[-1, 500])

    k = Arrhenius(1, 1000, 298, Tmin=100, Tmax=500, name='test')
    with pytest.raises(RangeError):
        _ = k(-300)
    with pytest.raises(RangeError):
        _ = k(-1, 'K')
    _ = k(550, 'K')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))


def test_evaluation_Arrhenius():
    k0 = [1., 1.]
    EaR = [2000., 4000.]
    T0 = 400
    k = Arrhenius(k0, EaR, T0, name='test')
    assert np.all((np.isclose(k(T0, 'K'), k0)))
    assert np.all(np.isclose(k.A, k.eval(np.inf)))
    k = Arrhenius(k0, EaR)
    k1 = k(300, 'K')
    k2 = k(600, 'K')
    assert (np.isclose(k2[1]/k1[0], 1))  # type:ignore


def test_product_Arrhenius_number():
    k1 = Arrhenius(1e2, 1e4, T0=340, Tmin=300, Tmax=380, name='k1')
    T = 350.
    number = 100
    k1_value = k1(T, 'K')
    # int left
    k2 = number*k1
    assert (np.isclose(k1_value*number, k2.eval(T)))
    # float left
    k2 = float(number)*k1
    assert (np.isclose(k1_value*number, k2.eval(T)))
    # int right
    k2 = k1*number
    assert (np.isclose(k1_value*number, k2.eval(T)))
    # float right
    k2 = k1*float(number)
    assert (np.isclose(k1_value*number, k2.eval(T)))


def test_product_Arrhenius_Arrhenius_scalar(capsys):
    k1 = Arrhenius(1e2, 1e4, T0=340, Tmin=300, Tmax=380, name='k1')
    k2 = Arrhenius(2e2, 5e3, T0=360, Tmin=320, Tmax=400, name='k2')
    T = 350.
    k1_value = k1(T, 'K')
    k2_value = k2(T, 'K')
    k3 = k1*k2
    k3_value = k3(T, 'K')
    assert (np.isclose(k3_value, k1_value*k2_value))
    _ = k3(390, 'K')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))
    _ = k3(310, 'K')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))


def test_division_Arrhenius_Arrhenius_scalar(capsys):
    k1 = Arrhenius(1e2, 1e4, T0=340, Tmin=300, Tmax=380, name='k1')
    k2 = Arrhenius(2e2, 5e3, T0=360, Tmin=320, Tmax=400, name='k2')
    T = 350.
    k1_value = k1(T, 'K')
    k2_value = k2(T, 'K')
    k3 = k1/k2
    k3_value = k3(T, 'K')
    assert (np.isclose(k3_value, k1_value/k2_value))
    _ = k3(390, 'K')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))
    _ = k3(310, 'K')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))


def test_product_Arrhenius_Arrhenius_array(capsys):
    k1 = Arrhenius([1e2, 2e3], [2e4, 1e4], T0=[340, 341],
                   Tmin=[300, 301], Tmax=[380, 381], name='k1')
    k2 = Arrhenius([2e2, 3e3], [3e4, 4e4], T0=[350, 351],
                   Tmin=[310, 311], Tmax=[390, 391], name='k2')
    T = 350.
    k1_value = k1(T, 'K')
    k2_value = k2(T, 'K')
    k3 = k1*k2
    k3_value = k3(T, 'K')
    assert (np.all(np.isclose(k3_value, k1_value*k2_value)))
    _ = k3(390, 'K')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))
    _ = k3(310, 'K')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))


def test_division_Arrhenius_Arrhenius_array(capsys):
    k1 = Arrhenius([1e2, 2e3], [2e4, 1e4], T0=[340, 341],
                   Tmin=[300, 301], Tmax=[380, 381], name='k1')
    k2 = Arrhenius([2e2, 3e3], [3e4, 4e4], T0=[350, 351],
                   Tmin=[310, 311], Tmax=[390, 391], name='k2')
    T = 350.
    k1_value = k1(T, 'K')
    k2_value = k2(T, 'K')
    k3 = k1/k2
    k3_value = k3(T, 'K')
    assert (np.all(np.isclose(k3_value, k1_value/k2_value)))
    _ = k3(390, 'K')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))
    _ = k3(310, 'K')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))


def test_evaluation_Eyring():
    DSa = [1e2, 0]
    DHa = [0, 73.5e3]
    kappa = [0.5, 1]
    Tref = 300.
    k = Eyring(DSa, DHa, kappa, name='test')
    assert np.all(
        np.isclose(k(2*Tref, 'K')[0]/k(Tref, 'K')[0], 2))  # type: ignore
    assert (np.isclose(k(Tref, 'K')[-1], 1, rtol=0.1))  # type: ignore


def test_evaluation_TerminationCompositeModel():
    T0 = 298
    icrit = 50
    aS = 0.5
    aL = 0.2
    kt11 = Arrhenius(1, 2000, T0, name='kt11')
    kt = TerminationCompositeModel(kt11, icrit, aS, aL, 'kt')
    assert (np.isclose(kt.eval(T0, icrit, icrit), kt11.eval(T0)/icrit**aS))
    assert len(kt.eval(T0, np.arange(1, 1000, 1),
               np.arange(1, 1000, 1)))  # type: ignore


def test_evaluation_PropagationHalfLength():
    T0 = 298
    ihalf = 2
    C = 11
    kp = Arrhenius(1e3, 2000, T0=298., name='kp(inf)')
    kpi = PropagationHalfLength(kp, C, ihalf, name='kp(i)')
    assert (np.isclose(kpi.eval(T0, 1)/kp.eval(T0), C))
    assert (np.isclose(kpi.eval(T0, ihalf+1)/kp.eval(T0), (C+1)/2))
    assert len(kpi.eval(T0, np.arange(1, 101)))  # type: ignore

# %% DIPPR


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
