# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
import pytest

from polykin.utils.exceptions import ShapeError
from polykin.utils.math import (
    add_dicts,
    convert_FloatOrVectorLike_to_FloatOrVector,
    convert_FloatOrVectorLike_to_FloatVector,
)


def test_add_dicts():
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 3, "c": 4}
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
    a = [1, [1, 2], np.array([3, 4, 5])]
    result = convert_FloatOrVectorLike_to_FloatVector(a, equal_shapes=False)
    assert isinstance(result, list)
    assert all(isinstance(x, np.ndarray) for x in result)
    with pytest.raises(ShapeError):
        _ = convert_FloatOrVectorLike_to_FloatVector(a, equal_shapes=True)
    # Input with indentical shapes
    a = [[1, 2], np.array([3, 4])]
    for equal_shapes in [False, True]:
        result = convert_FloatOrVectorLike_to_FloatVector(a, equal_shapes=False)
        assert isinstance(result, list)
        assert all(isinstance(x, np.ndarray) for x in result)


def test_convert_FloatOrVectorLike_to_FloatOrVector():
    # Input with different shapes
    a = [1, 2.0, [1, 2], np.array([3, 4, 5])]
    result = convert_FloatOrVectorLike_to_FloatOrVector(a)
    assert isinstance(result, list)
    assert isinstance(result[0], int)
    assert isinstance(result[1], float)
    assert isinstance(result[2], np.ndarray)
    assert isinstance(result[3], np.ndarray)
