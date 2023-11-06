# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023


from typing import Union, Any
from nptyping import NDArray, Shape, Int32, Float64


IntArray = NDArray[Any, Int32]
IntArrayLike = Union[list[int], IntArray]
IntOrArray = Union[int, IntArray]
IntOrArrayLike = Union[int, IntArrayLike]

FloatArray = NDArray[Any, Float64]
FloatArrayLike = Union[list[float], FloatArray]
FloatOrArray = Union[float, FloatArray]
FloatOrArrayLike = Union[float, FloatArrayLike]

FloatVector = NDArray[Shape['*'], Float64]
FloatVectorLike = Union[list[float], FloatVector]
FloatOrVector = Union[float, FloatVector]
FloatOrVectorLike = Union[float, FloatVectorLike]

FloatMatrix = NDArray[Shape['*, *'], Float64]
FloatSquareMatrix = NDArray[Shape['Dim, Dim'], Float64]

FloatRangeArray = NDArray[Shape['2'], Float64]
