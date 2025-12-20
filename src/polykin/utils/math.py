# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""
Mathematical and data-handling utilities used throughout PolyKin.

This module provides small helper functions and constants for numerical work,
including dictionary arithmetic, NumPy-based vectorization of instance methods,
conversion of scalar- and array-like inputs to standardized NumPy formats, and
basic linear-algebra utilities. The functionality is primarily intended to
support polymerization kinetics calculations while keeping type handling and
array shapes consistent across the library.
"""

import functools
from typing import Any

import numpy as np

from .tools import check_shapes
from .types import (
    FloatOrArray,
    FloatOrArrayLike,
    FloatOrVector,
    FloatOrVectorLike,
    FloatSquareMatrix,
    FloatVector,
)

_finfo = np.finfo(np.float64)
eps = float(_finfo.eps)
huge = float(_finfo.max)
tiny = float(_finfo.tiny)


def add_dicts(
    d1: dict[Any, int | float],
    d2: dict[Any, int | float],
    *,
    new: bool = True,
) -> dict[Any, int | float]:
    r"""Add two dictionaries by summing the values for the same key.

    Parameters
    ----------
    d1 : dict[Any, int | float]
        first dictionary
    d2 : dict[Any, int | float]
        second dictionary
    new : bool
        if True, a new dictionary will be created (`d = d1 + d2`), otherwise,
        d1 will be modified in place (`d1 <- d1 + d2`).

    Returns
    -------
    dict[Any, int | float]
        Sum of both dictionaries.
    """
    if new:
        dout = d1.copy()
    else:
        dout = d1

    for key, value in d2.items():
        dout[key] = dout.get(key, 0.0) + value
    return dout


class vectorize(np.vectorize):
    """Vectorize decorator for instance methods.

    Notes
    -----
    `vectorize` provides convenience, not performance. The wrapped function
    is still called element by element.
    """

    def __get__(self, obj, objtype):
        """Bind the vectorized function to an instance method."""
        return functools.partial(self.__call__, obj)


def convert_FloatOrArrayLike_to_FloatOrArray(
    a: list[FloatOrArrayLike],
) -> list[FloatOrArray]:
    """Convert list of `FloatOrArrayLike` to list of `FloatOrArray`."""
    result = []
    for item in a:
        if isinstance(item, (list, tuple)):
            item = np.array(item, dtype=np.float64)
        result.append(item)
    return result


def convert_FloatOrVectorLike_to_FloatOrVector(
    a: list[FloatOrVectorLike],
) -> list[FloatOrVector]:
    """Convert list of `FloatOrVectorLike` to list of `FloatOrVector`."""
    result = []
    for item in a:
        if isinstance(item, (list, tuple)):
            item = np.array(item, dtype=np.float64)
        result.append(item)
    return result


def convert_FloatOrVectorLike_to_FloatVector(
    a: list[FloatOrVectorLike],
    equal_shapes: bool = True,
) -> list[FloatVector]:
    """Convert list of `FloatOrVectorLike` to list of `FloatVector`."""
    result = []
    for item in a:
        if np.isscalar(item):
            item = (item,)
        if isinstance(item, (list, tuple)):
            item = np.array(item, dtype=np.float64)
        result.append(item)
    if equal_shapes:
        check_shapes(result)
    return result


def enforce_symmetry(matrix: FloatSquareMatrix) -> None:
    r"""Make a matrix symmetrical based on its upper triangle.

    Parameters
    ----------
    matrix : FloatSquareMatrix
        Matrix to be transformed in-place.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")

    N = matrix.shape[0]
    mask = np.tril(np.ones((N, N), dtype=bool), k=-1)
    matrix[mask] = matrix.T[mask]
