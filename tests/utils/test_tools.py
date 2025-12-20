# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024


from numbers import Number

import numpy as np
import pytest
from numpy import isclose

from polykin.utils.exceptions import RangeError
from polykin.utils.tools import (
    check_subclass,
    check_type,
    convert_check_pressure,
    pprint_matrix,
)


def test_check_type():
    # Test with correct types
    check_type(5, int, "a")
    check_type(3.14, float, "a")
    check_type([1, 2, 3], list, "a")
    check_type((1, 2), tuple, "a")
    check_type({"a": 1}, dict, "a")
    check_type(np.array([1, 2]), np.ndarray, "a")

    # Test with incorrect types
    with pytest.raises(TypeError):
        check_type(5, float, "a")
    with pytest.raises(TypeError):
        check_type(3.14, int, "a")
    with pytest.raises(TypeError):
        check_type([1, 2, 3], tuple, "a")
    with pytest.raises(TypeError):
        check_type((1, 2), list, "a")
    with pytest.raises(TypeError):
        check_type({"a": 1}, list, "a")
    with pytest.raises(TypeError):
        check_type(np.array([1, 2]), list, "a")

    # Test with containers
    check_type([1, 2, 3], int, "a", check_inside=True)
    check_type((1.0, 2.0, 3.0), float, "a", check_inside=True)
    check_type({"a", "b", "c"}, str, "a", check_inside=True)
    with pytest.raises(TypeError):
        check_type([1, "2", 3], int, "a", check_inside=True)


def test_check_subclass():
    check_subclass(int, object, "a")
    check_subclass(float, Number, "a")
    check_subclass([int, float, complex], Number, "a", check_inside=True)
    with pytest.raises(TypeError):
        check_subclass(str, Number, "a")
    with pytest.raises(TypeError):
        check_subclass([int, str, float], Number, "a", check_inside=True)


def test_pprint_matrix():
    matrix = np.array([[10.2552, 2], [3.456789, 4.567890]])
    expected_output = "[[1.03e+01 2.00e+00]\n [3.46e+00 4.57e+00]]\n"
    assert pprint_matrix(matrix) == expected_output


def test_convert_check_pressure():
    for Psol, unit in zip([1, 1e5, 1e6], ["Pa", "bar", "MPa"]):
        P = convert_check_pressure(1.0, unit)
        assert isclose(P, Psol)
    with pytest.raises(RangeError):
        convert_check_pressure(-1.0, "bar")
