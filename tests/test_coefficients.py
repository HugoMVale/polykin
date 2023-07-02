from polykin import Arrhenius, Eyring, CompositeModelTermination
from polykin.utils import RangeWarning, RangeError

import pytest
import numpy as np


def test_input_validation():
    with pytest.raises(ValueError):
        _ = Arrhenius(-1, 1, 1)
    with pytest.raises(ValueError):
        _ = Arrhenius(1, -1, 1)
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
        _ = Arrhenius([1, -2], [2, 3], [500, 600])
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [2, -3], [500, 600])
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
    with pytest.warns(RangeWarning):
        _ = k(550, 'K')


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


def test_product_Arrhenius_scalar():
    k1 = Arrhenius(1e2, 1e4, T0=340, Tmin=300, Tmax=380, name='k1')
    k2 = Arrhenius(2e2, 5e3, T0=360, Tmin=320, Tmax=400, name='k2')
    T = 350.
    k1_value = k1(T, 'K')
    k2_value = k2(T, 'K')
    k3 = k1*k2
    k3_value = k3(T, 'K')
    assert (np.isclose(k3_value, k1_value*k2_value))
    with pytest.warns(RangeWarning):
        _ = k3(390, 'K')
    with pytest.warns(RangeWarning):
        _ = k3(310, 'K')


def test_product_Arrhenius_array():
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
    with pytest.warns(RangeWarning):
        _ = k3(390, 'K')
    with pytest.warns(RangeWarning):
        _ = k3(310, 'K')


def test_evaluation_Eyring():
    DSa = [1e2, 0]
    DHa = [0, 73.5e3]
    kappa = [0.5, 1]
    Tref = 300.
    k = Eyring(DSa, DHa, kappa, name='test')
    assert np.all((np.isclose(k(2*Tref, 'K')[0]/k(Tref, 'K')[0], 2)))
    assert (np.isclose(k(Tref, 'K')[-1], 1, rtol=0.1))


def test_evaluation_TerminationCompostiteModel():
    T0 = 298
    icrit = 50
    aS = 0.5
    aL = 0.2
    kt11 = Arrhenius(1, 2000, T0, name='kt11')
    kt = CompositeModelTermination(kt11, icrit, aS, aL, 'kt')
    assert (np.isclose(kt.eval(T0, icrit, icrit), kt11.eval(T0)/icrit**aS))
    assert len(kt.eval(T0, np.arange(1, 1000, 1), np.arange(1, 1000, 1)))
