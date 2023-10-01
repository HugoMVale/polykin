# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.physprops.tait import Tait
from polykin.utils import RangeError


import pytest
import numpy as np


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
        name="Example Handbook Polymer Solution Thermodynamics, p.39"
    )


atol = 0e0
rtol = 1e-4


def test_input_validation(tait_instance, capsys):
    with pytest.raises(ValueError):
        _ = Tait(1., 3e-7, 7e-10, 2e8, 4e-3)
    with pytest.raises(ValueError):
        _ = Tait(8e-4, 6e4, 7e-10, 2e8, 4e-3)
    with pytest.raises(ValueError):
        _ = Tait(8e-4, 3e-7, 1e-6, 2e8, 4e-3)
    with pytest.raises(ValueError):
        _ = Tait(8e-4, 3e-7, 7e-10, 178, 4e-3)
    with pytest.raises(ValueError):
        _ = Tait(8e-4, 3e-7, 7e-10, 2e8, 1)
    with pytest.raises(RangeError):
        _ = tait_instance.__call__(-1, 1, Tunit='K')
    with pytest.raises(RangeError):
        _ = tait_instance.__call__(1, -1, Tunit='K')


def test_Trange_warning(tait_instance, capsys):
    _ = tait_instance.__call__(450, 10, Tunit='K')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))


def test_Prange_warning(tait_instance, capsys):
    _ = tait_instance.__call__(400, 2010, Tunit='K', Punit='bar')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))


def test_repr(tait_instance):
    out = tait_instance.__repr__()
    assert out.startswith('name')


def test_V0(tait_instance):
    V0 = tait_instance.eval(432.15, 0)
    assert np.isclose(V0, 8.9019e-4, atol=atol, rtol=rtol)


def test_B(tait_instance):
    B = tait_instance._B(432.15)
    assert np.isclose(B, 1.4855e8, atol=atol, rtol=rtol)


def test_V_eval(tait_instance):
    V = tait_instance.eval(432.15, 2e8)
    assert np.isclose(V, 8.2232e-4, atol=atol, rtol=rtol)


def test_V_call(tait_instance):
    V = tait_instance(159., 2000, Tunit='C', Punit='bar')
    assert np.isclose(V, 8.2232e-4, atol=atol, rtol=rtol)


def test_alpha(tait_instance):
    alpha = tait_instance.alpha(432.15, 2e8)
    assert np.isclose(alpha, 3.5012e-4, atol=atol, rtol=rtol)


def test_beta(tait_instance):
    beta = tait_instance.beta(432.15, 2e8)
    assert np.isclose(beta, 2.7765e-10, atol=atol, rtol=rtol)
