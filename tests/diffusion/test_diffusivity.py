# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.diffusion.vrentasduda import VrentasDudaBinary
from polykin.diffusion.estimation_methods import wilke_chang, hayduk_minhas, \
    wilke_lee

import numpy as np
import pytest

# %% VrentasDudaBinary


@pytest.fixture
def vrentas_instance():
    return VrentasDudaBinary(
        D0=4.82e-4,
        E=0.,
        V1star=0.917,
        V2star=0.850,
        z=0.58,
        K11=1.45e-3,
        K12=5.82e-4,
        K21=-86.32,
        K22=-327.0,
        Tg1=0.0,
        Tg2=0.0,
        y=1.0,
        X=0.354,
        name="toluene/polystyrene in Handbook of Diffusion..., page 123."
    )


def test_repr(vrentas_instance):
    out = vrentas_instance.__repr__()
    assert out.startswith('name')


def test_selfd(vrentas_instance):
    D1 = vrentas_instance.selfd(0.2, 308.)
    assert (np.isclose(D1, 1.435e-8, atol=0, rtol=1e-3))


def test_mutual(vrentas_instance):
    D = vrentas_instance.mutual(0.2, 308.)
    assert (np.isclose(D, 7.884e-9, atol=0, rtol=1e-3))


def test_call(vrentas_instance):
    w1 = 0.5
    T = 298.15
    D1 = vrentas_instance.selfd(w1, T)
    D = vrentas_instance.mutual(w1, T)

    assert (np.isclose(D1, vrentas_instance([w1], T, 'K', selfd=True)[0]))
    assert (np.isclose(D, vrentas_instance(w1, T, 'K')))
    assert (np.isclose(D1, vrentas_instance(w1, 25., 'C', selfd=True)))
    assert (np.isclose(D, vrentas_instance(w1, 25., 'C')))


def test_plot(vrentas_instance):
    out = vrentas_instance.plot(T=25., return_objects=True)
    assert len(out) == 2

# %% Infinite dilution eqquations


def test_wilke_chang():
    "Ethylbenze in water. Example 11.5, page 599 of Reid et al."
    result = wilke_chang(T=293.,
                         MA=106.17e-3,
                         MB=18.0e-3,
                         rhoA=761.,
                         viscB=1e-3,
                         phi=2.6
                         )
    assert (np.isclose(result, 0.77e-9, atol=0, rtol=1e-2))


def test_hayduk_minhas_1():
    "Hexane in hexane."
    result = hayduk_minhas(T=298.15,
                           method='paraffin',
                           MA=86.2e-3,
                           rhoA=613.,
                           viscB=0.298e-3,
                           )
    assert (np.isclose(result, 4.2e-9, atol=0, rtol=3e-2))


def test_hayduk_minhas_2():
    "Ethylbenze in water"
    result = hayduk_minhas(T=293.,
                           method='aqueous',
                           MA=106.17e-3,
                           rhoA=761.,
                           viscB=1e-3,
                           )
    assert (np.isclose(result, 6.97e-10, atol=0, rtol=1e-2))


def test_hayduk_minhas_input():
    with pytest.raises(ValueError):
        _ = hayduk_minhas(T=293.,
                          method='nomethod',
                          MA=106.17e-3,
                          rhoA=761.,
                          viscB=1e-3,
                          )

# %% Gas-phase


def test_wilke_lee_air():
    "Allyl chloride in air. Example 11.3, page 589 of Reid et al."
    result = wilke_lee(T=298.,
                       P=1e5,
                       MA=76.5e-3,
                       MB=29.0e-3,
                       rhoA=1e3*76.5/87.5,
                       rhoB=None,
                       TA=318.3,
                       TB=None
                       )
    assert (np.isclose(result, 0.10e-4, atol=0, rtol=1e-2))


def test_wilke_lee():
    "Allyl chloride in steam."
    result = wilke_lee(T=298.,
                       P=1e5,
                       MA=76.5e-3,
                       MB=18.0e-3,
                       rhoA=1e3*76.5/87.5,
                       rhoB=1000.,
                       TA=318.3,
                       TB=373.
                       )
    assert (np.isclose(result, 0.10e-4, atol=0, rtol=1e-1))


def test_wilke_lee_input():
    with pytest.raises(ValueError):
        _ = wilke_lee(T=298.,
                      P=1e5,
                      MA=76.5e-3,
                      MB=29.0e-3,
                      rhoA=1e3*76.5/87.5,
                      rhoB=None,  # issue
                      TA=318.3,
                      TB=300.  # issue
                      )
    with pytest.raises(ValueError):
        _ = wilke_lee(T=298.,
                      P=1e5,
                      MA=76.5e-3,
                      MB=29.0e-3,
                      rhoA=1e3*76.5/87.5,
                      rhoB=1000.,  # issue
                      TA=318.3,
                      TB=None  # issue
                      )
