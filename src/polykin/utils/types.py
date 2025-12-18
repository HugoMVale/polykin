# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""
Type aliases for numeric scalars and NumPy arrays used throughout the package.

This module defines common scalar, array-like, vector, and matrix type aliases
based on ``numpy.typing.NDArray``. The aliases primarily constrain data types
(e.g., ``float64`` or integer) and provide consistent terminology across the
codebase.

Array shapes (such as vectors or matrices) are documented by convention and
validated at runtime where required, as NumPy's static typing currently does
not enforce shape information.
"""

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

# Numeric type variables
Number = TypeVar("Number", float, complex)
"""A generic numeric type, either `float` or `complex`."""

Floaty = TypeVar("Floaty", float, NDArray[np.float64])
"""A type that can be either a float scalar or a NumPy array of float64."""

# Integer arrays and sequences
IntArray = NDArray[np.int_]
"""A NumPy array with integer dtype."""

IntArrayLike = list[int] | tuple[int, ...] | IntArray
"""
Any object that can be interpreted as an integer array, including:

- Python list of ints
- Python tuple of ints
- NumPy integer array
"""

IntVector = NDArray[np.int_]
"""A 1-dimensional NumPy array of integers."""

IntVectorLike = list[int] | tuple[int, ...] | IntVector
"""
Any object that can be interpreted as a 1D integer vector, including:

- Python list of ints
- Python tuple of ints
- 1D NumPy integer array
"""

# Float arrays and sequences
FloatArray = NDArray[np.float64]
"""A NumPy array with float64 dtype."""

FloatArrayLike = list[float] | tuple[float, ...] | FloatArray
"""
Any object that can be interpreted as a float array, including:

- Python list of floats
- Python tuple of floats
- NumPy float64 array
"""

FloatOrArray = float | FloatArray
"""A float scalar or a NumPy float64 array."""

FloatOrArrayLike = float | FloatArrayLike
"""A float scalar or any array-like object convertible to a float array."""

FloatVector = NDArray[np.float64]
"""A 1-dimensional NumPy float64 array."""

FloatVectorLike = list[float] | tuple[float, ...] | FloatVector
"""
Any object that can be interpreted as a 1D float vector, including:

- Python list of floats
- Python tuple of floats
- 1D NumPy float64 array
"""

FloatOrVector = float | FloatVector
"""A float scalar or a 1D float64 array."""

FloatOrVectorLike = float | FloatVectorLike
"""A float scalar or any array-like object convertible to a 1D float vector."""

# Float matrices
FloatMatrix = NDArray[np.float64]
"""A 2-dimensional NumPy float64 array of arbitrary shape."""

FloatSquareMatrix = NDArray[np.float64]
"""A 2-dimensional NumPy float64 array with equal number of rows and columns."""

Float2x2Matrix = NDArray[np.float64]
"""A NumPy float64 array of shape (2, 2)."""

# Specialized float arrays
FloatRangeArray = NDArray[np.float64]
"""A NumPy float64 array of shape (2,) representing a range of values."""
