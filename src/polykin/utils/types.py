# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023


from typing import Any, TypeVar

import numpy as np
from nptyping import Float64, Int32, NDArray, Shape

Number = TypeVar("Number", float, complex)

IntArray = np.ndarray[Any, np.dtype[np.int32]]
IntArrayLike = list[int] | tuple[int, ...] | IntArray

IntVector = NDArray[Shape['*'], Int32]
IntVectorLike = list[int] | tuple[int, ...] | IntVector

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
FloatArrayLike = list[float] | tuple[float, ...] | FloatArray
FloatOrArray = float | FloatArray
FloatOrArrayLike = float | FloatArrayLike

FloatVector = NDArray[Shape['*'], Float64]
FloatVectorLike = list[float] | tuple[float, ...] | FloatVector
FloatOrVector = float | FloatVector
FloatOrVectorLike = float | FloatVectorLike

FloatMatrix = NDArray[Shape['*, *'], Float64]
FloatSquareMatrix = NDArray[Shape['Dim, Dim'], Float64]
Float2x2Matrix = NDArray[Shape['2, 2'], Float64]

FloatRangeArray = NDArray[Shape['2'], Float64]
