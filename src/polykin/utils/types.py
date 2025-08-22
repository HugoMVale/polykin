# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023


from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

Number = TypeVar("Number", float, complex)
Floaty = TypeVar("Floaty", float, NDArray[np.float64])

IntArray = NDArray[np.int_]
IntArrayLike = list[int] | tuple[int, ...] | IntArray

IntVector = NDArray[np.int_]
IntVectorLike = list[int] | tuple[int, ...] | IntVector

FloatArray = NDArray[np.float64]
FloatArrayLike = list[float] | tuple[float, ...] | FloatArray
FloatOrArray = float | FloatArray
FloatOrArrayLike = float | FloatArrayLike

FloatVector = NDArray[np.float64]
FloatVectorLike = list[float] | tuple[float, ...] | FloatVector
FloatOrVector = float | FloatVector
FloatOrVectorLike = float | FloatVectorLike

FloatMatrix = NDArray[np.float64]
FloatSquareMatrix = NDArray[np.float64]
Float2x2Matrix = NDArray[np.float64]

FloatRangeArray = NDArray[np.float64]
