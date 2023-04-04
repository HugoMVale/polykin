from polykin import Arrhenius, Eyring, TerminationCompositeModel

import pytest
import numpy as np


def test_input_validation():
    with pytest.raises(ValueError):
        _ = Arrhenius(-1, 1, 1)
    with pytest.raises(ValueError):
        _ = Arrhenius(1, -1, 1)
    with pytest.raises(ValueError):
        _ = Arrhenius(1, -1, -1)
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [1, 2, 3], 298)
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [2, 3], [200])
    with pytest.raises(ValueError):
        _ = Arrhenius([1, -2], [2, 3], [500, 600])
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [2, -3], [500, 600])
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [2, 3], [500, -100])
    with pytest.raises(ValueError):
        _ = Eyring([-1e3, 2e3], [1e3, 2e3], [0.5, 1])
    with pytest.raises(ValueError):
        _ = Eyring([1e3, 2e3], [-1e3, 2e3], [0.5, 1])
    with pytest.raises(ValueError):
        _ = Eyring([1e3, 2e4], [1e3, 2e3], [2, .5])
    with pytest.raises(ValueError):
        _ = Eyring([1e3, 2e4], [1e3, 2e3], [0.4, .5, 0.8])

    k = Arrhenius(1, 1000, 298, 'test')
    with pytest.raises(ValueError):
        _ = k(-300)
    with pytest.raises(ValueError):
        _ = k(-1, kelvin=True)


def test_evaluation_Arrhenius():
    k0 = [1, 1]
    EaR = [2000, 4000]
    T0 = 400
    k = Arrhenius(k0, EaR, T0, 'test')
    assert np.all((np.isclose(k(T0, True), k0)))
    k = Arrhenius(k0, EaR)
    k1 = k(300, True)
    k2 = k(600, True)
    assert (np.isclose(k2[1]/k1[0], 1))  # type:ignore


def test_evaluation_Eyring():
    DSa = [1e2, 0]
    DHa = [0, 73.5e3]
    kappa = [0.5, 1]
    Tref = 300
    k = Eyring(DSa, DHa, kappa, 'test')
    assert np.all((np.isclose(k(2*Tref, True)[0]/k(Tref, True)[0], 2)))
    assert (np.isclose(k(Tref, True)[-1], 1, rtol=0.1))


def test_evaluation_TerminationCompostiteModel():
    T0 = 298
    icrit = 50
    aS = 0.5
    aL = 0.2
    kt11 = Arrhenius(1, 2000, T0, 'kt11')
    kt = TerminationCompositeModel(kt11, icrit, aS, aL, 'kt')
    assert (np.isclose(kt.eval(T0, icrit, icrit), kt11.eval(T0)/icrit**aS))
    assert len(kt.eval(T0, np.arange(1, 1000, 1), np.arange(1, 1000, 1)))
