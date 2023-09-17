from polykin.diffusion.diffusivity import VrentasDudaBinary

import numpy as np
import pytest


# Define refernce input values for testing
test_params = {
    'D0': 4.82e-4,
    'E': 0.,
    'V1star': 0.917,
    'V2star': 0.850,
    'z': 0.58,
    'K11': 1.45e-3,
    'K12': 5.82e-4,
    'K21': -86.32,
    'K22': -327.0,
    'Tg1': 0.0,
    'Tg2': 0.0,
    'y': 1.0,
    'X': 0.354,
    'name': 'toluene/polystyrene "Handbook of Diffusion...", page 123.'
}


@pytest.fixture
def vrentas_instance():
    return VrentasDudaBinary(**test_params)


def test_selfd(vrentas_instance):
    D1 = vrentas_instance.selfd(0.2, 308.)
    assert (np.isclose(D1, 1.435e-8, rtol=1e-3))


def test_mutual(vrentas_instance):
    D = vrentas_instance.mutual(0.2, 308.)
    assert (np.isclose(D, 7.884e-9, rtol=1e-3))


def test_call(vrentas_instance):
    w1 = 0.5
    T = 298.15
    D1 = vrentas_instance.selfd(w1, T)
    D = vrentas_instance.mutual(w1, T)

    assert (np.isclose(D1, vrentas_instance(w1, T, 'K', True)))
    assert (np.isclose(D, vrentas_instance(w1, T, 'K')))
    assert (np.isclose(D1, vrentas_instance(w1, 25., selfd=True)))
    assert (np.isclose(D, vrentas_instance(w1, 25.)))
