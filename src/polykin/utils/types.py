# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023


from typing import Any, TypeVar, Union

from nptyping import Float64, Int32, NDArray, Shape

Number = TypeVar("Number", float, complex)

IntArray = NDArray[Any, Int32]
IntArrayLike = Union[list[int], tuple[int, ...], IntArray]

IntVector = NDArray[Shape['*'], Int32]
IntVectorLike = Union[list[int], tuple[int, ...], IntVector]

FloatArray = NDArray[Any, Float64]
FloatArrayLike = Union[list[float], tuple[float, ...], FloatArray]
FloatOrArray = Union[float, FloatArray]
FloatOrArrayLike = Union[float, FloatArrayLike]

FloatVector = NDArray[Shape['*'], Float64]
FloatVectorLike = Union[list[float], tuple[float, ...], FloatVector]
FloatOrVector = Union[float, FloatVector]
FloatOrVectorLike = Union[float, FloatVectorLike]

FloatMatrix = NDArray[Shape['*, *'], Float64]
FloatSquareMatrix = NDArray[Shape['Dim, Dim'], Float64]
Float2x2Matrix = NDArray[Shape['2, 2'], Float64]

FloatRangeArray = NDArray[Shape['2'], Float64]
