# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
import pytest

from polykin.properties.pvt_polymer import (Flory, HartmannHaque,
                                            SanchezLacombe, Tait)
from polykin.utils.exceptions import RangeError

ATOL = 0e0
RTOL = 1e-4

# %% Tait


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


def test_Tait_input_validation(tait_instance, capsys):
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
        _ = tait_instance.V(-1, 1, Tunit='K')
    with pytest.raises(RangeError):
        _ = tait_instance.V(1, -1, Tunit='K')


def test_Tait_Trange_warning(tait_instance, capsys):
    _ = tait_instance.V(450, 10, Tunit='K')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))


def test_Tait_Prange_warning(tait_instance, capsys):
    _ = tait_instance.V(400, 2010, Tunit='K', Punit='bar')
    out, _ = capsys.readouterr()
    assert (out.lower().startswith('warning'))


def test_Tait_repr(tait_instance):
    out = tait_instance.__repr__()
    assert out.startswith('name')


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
    V = tait_instance.V(159., 2000, Tunit='C', Punit='bar')
    assert np.isclose(V, 8.2232e-4, atol=ATOL, rtol=RTOL)


def test_Tait_alpha(tait_instance):
    alpha = tait_instance.alpha(432.15, 2e8)
    assert np.isclose(alpha, 3.5012e-4, atol=ATOL, rtol=RTOL)


def test_Tait_beta(tait_instance):
    beta = tait_instance.beta(432.15, 2e8)
    assert np.isclose(beta, 2.7765e-10, atol=ATOL, rtol=RTOL)


# %% Flory


@pytest.fixture
def flory_instance():
    return Flory(
        0.9455e-3, 7396., 396e6,
        name="PIB, Handbook of diffusion and thermal properties.., , p. 93.")


def test_Flory_V(flory_instance):
    "Handbook of diffusion and thermal properties.., , p. 93."
    assert np.isclose(flory_instance.V(335., 70, Punit='MPa'),
                      1.0753e-3, atol=0, rtol=2e-4)


def test_Flory_alpha(flory_instance):
    "Handbook of diffusion and thermal properties.., , p. 93."
    assert np.isclose(flory_instance.alpha(335., 70e6),
                      4.2779e-4, atol=0, rtol=2e-3)


def test_Flory_beta():
    "Comparison against value from Tait."
    m = Flory(V0=0.7204e-3, T0=7717., P0=568.8e6, name="PMMA")
    assert np.isclose(m.beta(432.15, 2e8),
                      2.8e-10, atol=0, rtol=0.2)

# %% HartmannHaque


@pytest.fixture
def HartmannHaque_instance():
    return HartmannHaque(
        V0=0.9935e-3, T0=1422., P0=2976e6,
        name="PIB, Handbook of diffusion and thermal properties.., , p. 85.")


def test_HartmannHaque_V(HartmannHaque_instance):
    "PIB, Handbook of diffusion and thermal properties.., , p. 85."
    m = HartmannHaque_instance
    assert np.isclose(m.V(335., 70, Punit='MPa'),
                      1.0756e-3, atol=0, rtol=2e-4)


def test_HartmannHaque_alpha(HartmannHaque_instance):
    m = HartmannHaque_instance
    assert np.isclose(m.alpha(335., 70e6),
                      4.3e-4, atol=0, rtol=2e-2)

# %% SanchezLacombe


@pytest.fixture
def SanchezLacombe_instance():
    return SanchezLacombe(
        V0=1.0213e-3, T0=623., P0=350.4e6,
        name="PIB, Handbook of diffusion and thermal properties.., , p. 78.")


def SanchezLacombe_V(SanchezLacombe_instance):
    "PIB, Handbook of diffusion and thermal properties.., , p. 85."
    m = SanchezLacombe_instance
    assert np.isclose(m.V(335., 70, Punit='MPa'),
                      1.0747e-3, atol=0, rtol=2e-4)


def test_SanchezLacombe_alpha(SanchezLacombe_instance):
    m = SanchezLacombe_instance
    assert np.isclose(m.alpha(335., 70e6),
                      4.3e-4, atol=0, rtol=5e-2)


def test_databanks():
    for method in [Tait, Flory, HartmannHaque, SanchezLacombe]:
        table = method.get_database()
        polymers = table.index.to_list()
        for polymer in polymers:
            m = method.from_database(polymer)
            assert m is not None
            rhoP = 1/m.eval(298., 1e5)
            assert (rhoP > 750. and rhoP < 2300.)
