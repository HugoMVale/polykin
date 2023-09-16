from polykin.copolymerization import CopoData
from polykin.utils import RangeError, ShapeError

import pytest
import numpy as np

# Create a fixture for a valid instance of CopoData
@pytest.fixture
def valid_copo_data():
    f1 = [0.1, 0.2, 0.3]
    F1 = [0.4, 0.5, 0.6]
    return CopoData(M1="M1", M2="M2", f1=f1, F1=F1)


# Test initialization with valid data
def test_copo_data_initialization(valid_copo_data):
    assert valid_copo_data.M1 == "M1"
    assert valid_copo_data.M2 == "M2"
    assert np.allclose(valid_copo_data.f1, [0.1, 0.2, 0.3])
    assert np.allclose(valid_copo_data.F1, [0.4, 0.5, 0.6])
    assert valid_copo_data.sigma_f == 1e-3
    assert valid_copo_data.sigma_F == 5e-2
    assert valid_copo_data.name == ""
    assert valid_copo_data.reference == ""


# Test initialization with empty M1 or M2 (should raise ValueError)
def test_copo_data_empty_M1_M2():
    with pytest.raises(ValueError):
        CopoData(M1="", M2="M2", f1=[0.1, 0.2, 0.3], F1=[0.4, 0.5, 0.6])

    with pytest.raises(ValueError):
        CopoData(M1="M1", M2="", f1=[0.1, 0.2, 0.3], F1=[0.4, 0.5, 0.6])


# Test initialization with M1 and M2 being the same (should raise ValueError)
def test_copo_data_same_M1_M2():
    with pytest.raises(ValueError):
        CopoData(M1="M1", M2="M1", f1=[0.1, 0.2, 0.3], F1=[0.4, 0.5, 0.6])


# Test initialization with f1 and F1 having different lengths (should raise
# ShapeError)
def test_copo_data_different_lengths():
    with pytest.raises(ShapeError):
        CopoData(M1="M1", M2="M2", f1=[0.1, 0.2, 0.3], F1=[0.4, 0.5])

    with pytest.raises(ShapeError):
        CopoData(M1="M1", M2="M2", f1=[0.1, 0.2], F1=[0.4, 0.5, 0.6])


# Test initialization with invalid values for f1, F1, sigma_f, and sigma_F
# (out of bounds)
def test_copo_data_invalid_values():
    f1 = [1.5, 0.2, 0.3]
    F1 = [0.4, -0.2, 0.6]
    sigma_f = [0.1, 0.2, 0.3, 1.2]
    sigma_F = [-0.1, 0.2, 0.3, 0.9]

    with pytest.raises(RangeError):
        CopoData(M1="M1", M2="M2", f1=f1, F1=F1)

    with pytest.raises(RangeError):
        CopoData(M1="M1", M2="M2", f1=[0.1, 0.2, 0.3], F1=[0.4, 0.5, 0.6],
                 sigma_f=sigma_f)

    with pytest.raises(RangeError):
        CopoData(M1="M1", M2="M2", f1=[0.1, 0.2, 0.3], F1=[0.4, 0.5, 0.6],
                 sigma_F=sigma_F)


# Test the representation output
def test_copo_data_representation(valid_copo_data):
    expected_repr = (
        "M1:        M1\n"
        "M2:        M2\n"
        "f1:        [0.1 0.2 0.3]\n"
        "F1:        [0.4 0.5 0.6]\n"
        "sigma_f:   0.001\n"
        "sigma_F:   0.05\n"
        "name:      \n"
        "reference: "
    )
    assert repr(valid_copo_data) == expected_repr