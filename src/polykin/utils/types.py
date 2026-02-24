# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""
Type aliases and variables for numeric scalars and NumPy arrays.

This module provides a centralized set of type hints used throughout PolyKin to ensure
consistency and improve code readability. It leverages `numpy.typing` for data type
validation and uses PEP 585-style shape annotations to provide informal hints about array
dimensions.

Note:
    While shapes (e.g., 1D vectors vs 2D matrices) are annotated in the type aliases, most
    static type checkers (like Mypy) do not yet strictly enforce NumPy array shapes.
"""

from typing import Literal, TypeAlias, TypeVar

try:
    # Python ≥ 3.12
    from typing import override
except ImportError:  # Python < 3.12
    from typing_extensions import override

import numpy as np
from numpy import dtype
from numpy.typing import NDArray

__all__ = [
    "Number",
    "Floaty",
    "IntArray",
    "IntArrayLike",
    "IntVector",
    "IntVectorLike",
    "FloatArray",
    "FloatArrayLike",
    "FloatOrArray",
    "FloatOrArrayLike",
    "FloatVector",
    "FloatVectorLike",
    "FloatOrVector",
    "FloatOrVectorLike",
    "FloatMatrix",
    "FloatSquareMatrix",
    "Float2x2Matrix",
    "FloatRangeArray",
    "override",
]

# --- Numeric Type Variables ---

Number = TypeVar("Number", float, complex)
"""A TypeVar bound to `float` or `complex` for generic numeric operations."""

Floaty = TypeVar("Floaty", float, NDArray[np.float64])
"""A TypeVar representing either a float scalar or a float64 NumPy array."""

# --- Integer Arrays and Sequences ---

IntArray = NDArray[np.int_]
"""A NumPy array of integers (any shape)."""

IntArrayLike: TypeAlias = list[int] | tuple[int, ...] | IntArray
"""An object convertible to an integer array (list, tuple, or NDArray)."""

IntVector: TypeAlias = np.ndarray[tuple[int], dtype[np.int_]]
"""A 1-dimensional NumPy array of integers."""

IntVectorLike: TypeAlias = list[int] | tuple[int, ...] | IntVector
"""An object convertible to a 1D integer vector."""

# --- Float Arrays and Sequences ---

FloatArray = NDArray[np.float64]
"""A NumPy array of float64 (any shape)."""

FloatArrayLike: TypeAlias = list[float] | tuple[float, ...] | FloatArray
"""An object convertible to a float64 array (list, tuple, or NDArray)."""

FloatOrArray: TypeAlias = float | FloatArray
"""A union of a float scalar or a float64 array."""

FloatOrArrayLike: TypeAlias = float | FloatArrayLike
"""A union of a float scalar or any object convertible to a float array."""

FloatVector: TypeAlias = np.ndarray[tuple[int], dtype[np.float64]]
"""A 1-dimensional NumPy array of float64."""

FloatVectorLike: TypeAlias = list[float] | tuple[float, ...] | FloatVector
"""An object convertible to a 1D float64 vector."""

FloatOrVector: TypeAlias = float | FloatVector
"""A union of a float scalar or a 1D float64 vector."""

FloatOrVectorLike: TypeAlias = float | FloatVectorLike
"""A union of a float scalar or any object convertible to a 1D float vector."""

# --- Float Matrices ---

FloatMatrix: TypeAlias = np.ndarray[tuple[int, int], dtype[np.float64]]
"""A 2-dimensional NumPy array of float64."""

FloatSquareMatrix: TypeAlias = np.ndarray[tuple[int, int], dtype[np.float64]]
"""A 2-dimensional float64 array with implicitly equal dimensions."""

Float2x2Matrix: TypeAlias = np.ndarray[tuple[Literal[2], Literal[2]], dtype[np.float64]]
"""A 2x2 NumPy array of float64."""

# --- Specialized Float Arrays ---

FloatRangeArray: TypeAlias = np.ndarray[tuple[Literal[2]], dtype[np.float64]]
"""A NumPy array of shape (2,) used to define [min, max] ranges."""
