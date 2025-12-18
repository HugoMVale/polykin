# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
import pytest

from polykin.properties.pvt import Tait
from polykin.utils.exceptions import RangeError

ATOL = 0e0
RTOL = 1e-4


@pytest.fixture
def tait_instance():
    return Tait(
        A0=8.2396e-4,
        A1=3.0490e-7,
        A2=7.0201e-10,
        B0=2.9803e8,
        B1=4.3789e-3,
        Tmin=387.15,
        Tmax=432.15,
        Pmin=0.1e6,
        Pmax=200e6,
        name="Example Handbook Polymer Solution Thermodynamics, p.39",
    )


def test_Tait_input_validation(tait_instance):
    with pytest.raises(ValueError):
        _ = Tait(1.0, 3e-7, 7e-10, 2e8, 4e-3)
    with pytest.raises(ValueError):
        _ = Tait(8e-4, 6e4, 7e-10, 2e8, 4e-3)
    with pytest.raises(ValueError):
        _ = Tait(8e-4, 3e-7, 1e-6, 2e8, 4e-3)
    with pytest.raises(ValueError):
        _ = Tait(8e-4, 3e-7, 7e-10, 178, 4e-3)
    with pytest.raises(ValueError):
        _ = Tait(8e-4, 3e-7, 7e-10, 2e8, 1)
    with pytest.raises(RangeError):
        _ = tait_instance.vs(-1, 1, Tunit="K")
    with pytest.raises(RangeError):
        _ = tait_instance.vs(1, -1, Tunit="K")


def test_Tait_Trange_warning(tait_instance):
    with pytest.warns(Warning):
        _ = tait_instance.vs(450, 10, Tunit="K")


def test_Tait_Prange_warning(tait_instance):
    with pytest.warns(Warning):
        _ = tait_instance.vs(400, 2010, Tunit="K", Punit="bar")


def test_Tait_repr(tait_instance):
    out = tait_instance.__repr__()
    assert out.startswith("name")


def test_Tait_V0(tait_instance):
    V0 = tait_instance.eval(432.15, 0)
    assert np.isclose(V0, 8.9019e-4, atol=ATOL, rtol=RTOL)


def test_Tait_B(tait_instance):
    B = tait_instance._B(432.15)
    assert np.isclose(B, 1.4855e8, atol=ATOL, rtol=RTOL)


def test_Tait_eval(tait_instance):
    V = tait_instance.eval(432.15, 2e8)
    assert np.isclose(V, 8.2232e-4, atol=ATOL, rtol=RTOL)


def test_Tait_V(tait_instance):
    vs = tait_instance.vs(159.0, 2000, Tunit="C", Punit="bar")
    assert np.isclose(vs, 8.2232e-4, atol=ATOL, rtol=RTOL)


def test_Tait_alpha(tait_instance):
    alpha = tait_instance.alpha(432.15, 2e8)
    assert np.isclose(alpha, 3.5012e-4, atol=ATOL, rtol=RTOL)


def test_Tait_beta(tait_instance):
    beta = tait_instance.beta(432.15, 2e8)
    assert np.isclose(beta, 2.7765e-10, atol=ATOL, rtol=RTOL)


def testTait__databank():
    table = Tait.get_database()
    polymers = table.index.to_list()
    for polymer in polymers:
        m = Tait.from_database(polymer)
        assert m is not None
        rhoP = 1 / m.eval(298.0, 1e5)
        assert rhoP > 750.0 and rhoP < 2.3e3
