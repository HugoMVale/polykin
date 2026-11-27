# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from typing import Any

import numpy as np
import pytest

from polykin.utils.exceptions import ShapeError
from polykin.utils.math import (
    add_dicts,
    convert_FloatOrArrayLike_to_FloatOrArray,
    convert_FloatOrVectorLike_to_FloatOrVector,
    convert_FloatOrVectorLike_to_FloatVector,
)


def test_add_dicts():
    d1: dict[Any, int | float] = {"a": 1, "b": 2}
    d2: dict[Any, int | float] = {"b": 3, "c": 4}
    result = add_dicts(d1, d2)
    expected = {"a": 1, "b": 5, "c": 4}
    assert result == expected
    assert id(result) != id(d1)
    assert id(result) != id(d2)
    result = add_dicts(d1, d2, new=False)
    assert d1 == expected
    assert id(result) == id(d1)


def test_convert_FloatOrVectorLike_to_FloatVector():
    # Input with different shapes
    a: Any = [1, [1, 2], np.array([3, 4, 5])]
    result = convert_FloatOrVectorLike_to_FloatVector(a, equal_shapes=False)
    assert isinstance(result, list)
    assert all(isinstance(x, np.ndarray) for x in result)
    assert all(x.dtype == np.float64 for x in result)
    with pytest.raises(ShapeError):
        _ = convert_FloatOrVectorLike_to_FloatVector(a, equal_shapes=True)
    # Input with indentical shapes
    a: Any = [[1, 2], np.array([3, 4])]
    for equal_shapes in [False, True]:
        result = convert_FloatOrVectorLike_to_FloatVector(a, equal_shapes=False)
        assert isinstance(result, list)
        assert all(isinstance(x, np.ndarray) for x in result)
        assert all(x.dtype == np.float64 for x in result)


def test_convert_FloatOrVectorLike_to_FloatOrVector():
    # Input with different shapes
    a: Any = [1, 2.0, [1, 2], np.array([3, 4, 5], dtype=np.int32)]
    result = convert_FloatOrVectorLike_to_FloatOrVector(a)
    assert isinstance(result, list)
    assert isinstance(result[0], int)
    assert isinstance(result[1], float)
    assert isinstance(result[2], np.ndarray)
    assert result[2].dtype == np.float64
    assert isinstance(result[3], np.ndarray)
    assert result[3].dtype == np.float64


def test_convert_FloatOrArrayLike_to_FloatOrArray():
    float64_array = np.array([1.0, 2.0])
    values: Any = [
        1.0,
        [1.0, 2.0],
        np.asarray([3, 4], dtype=np.int32),
        np.asarray([5, 6], dtype=np.float32),
        float64_array,
    ]
    result = convert_FloatOrArrayLike_to_FloatOrArray(values)

    assert result[0] == 1.0
    assert all(
        isinstance(item, np.ndarray) and item.dtype == np.float64 for item in result[1:]
    )
    assert result[-1] is float64_array


def test_convert_FloatOrVectorLike_to_FloatVector_normalizes_scalar_and_arrays():
    result = convert_FloatOrVectorLike_to_FloatVector(
        [1, np.array([2.0], dtype=np.float32)],  # type: ignore[list-item]
        equal_shapes=True,
    )

    assert all(item.dtype == np.float64 for item in result)
    assert all(item.shape == (1,) for item in result)
