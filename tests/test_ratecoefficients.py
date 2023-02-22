from polykin import Arrhenius, Eyring

import pytest
import numpy as np


def test_input_validation():
    with pytest.raises(ValueError):
        _ = Arrhenius(-1, 1, 1)
    with pytest.raises(ValueError):
        _ = Arrhenius(1, -1, 1)
    with pytest.raises(ValueError):
        _ = Arrhenius(1, -1, -300)
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [1, 2, 3], 300)
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [2, 3], [2])
    with pytest.raises(ValueError):
        _ = Arrhenius([1, -2], [2, 3], [5, 6])
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [2, -3], [5, 6])
    with pytest.raises(ValueError):
        _ = Arrhenius([1, 2], [2, 3], [5, -300])
    with pytest.raises(ValueError):
        _ = Eyring([-1e3, 2e3], [0.5, 1])
    with pytest.raises(ValueError):
        _ = Eyring([1e3, 2e4], [2, .5])
    with pytest.raises(ValueError):
        _ = Eyring([1e3, 2e4], [0.4, .5, 0.8])

    k = Arrhenius(1, 1000, 0, 'test')
    with pytest.raises(ValueError):
        _ = k(-300)
    with pytest.raises(ValueError):
        _ = k(-1, kelvin=True)


def test_evaluation_Arrhenius():
    k0 = [1, 1]
    EaR = [2000, 4000]
    T0 = 100
    k = Arrhenius(k0, EaR, T0, 'test')
    assert np.all((np.isclose(k(T0), k0)))
    k = Arrhenius(k0, EaR)
    k1 = k(300, True)
    k2 = k(600, True)
    assert (np.isclose(k2[1]/k1[0], 1))


def test_evaluation_Eyring():
    DGa = [0, 73.5e3]
    kappa = [0.5, 1]
    Tref = 300
    k = Eyring(DGa, kappa, 'test')
    assert np.all((np.isclose(k(2*Tref, True)[0]/k(Tref, True)[0], 2)))
    assert (np.isclose(k(Tref, True)[-1], 1, rtol=0.1))
