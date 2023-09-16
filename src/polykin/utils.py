from collections.abc import Iterable
import numbers
import functools
import numpy as np
from typing import Union, Any, Literal
from nptyping import NDArray, Shape, Int32, Float64

# %% Own types

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


# %% Maths

eps = float(np.finfo(np.float64).eps)

# %% Custom exceptions


class RangeWarning(Warning):
    pass


class RangeError(ValueError):
    pass


class ShapeError(ValueError):
    pass

# %% Check tools


def custom_error(var_name: str,
                 var_value: Any,
                 kind: type,
                 message: str = ""
                 ) -> None:
    """
    Custom error message function.

    Parameters
    ----------
    var_name : string
        Variable name.
    var_value : type(var_name)
        Value of variable `var_name`.
    kind : Error
        Type of Error (ValueError, TypeError, etc.).
    message : string, optional
        User message. The default is "".

    """
    # Print error message
    error_message = \
        f"`{var_name} = {var_value}` is not a valid input."
    # sys.tracebacklimit = 1
    if message != "":
        error_message += " " + message
    # print(f"{kind}: {error_message}")
    # sys.exit(1)
    raise kind(error_message)


def check_type(var_value: Any,
               valid_types: Union[type, tuple[type, ...]],
               var_name: str,
               check_inside: bool = False
               ) -> Any:
    """
    Check if a variable is of a given type.

    Parameters
    ----------
    var_value : type(var_name)
        Value of variable `var_name`.
    valid_types : Any
        Valid types of variable `var_name`
    var_name : str
        Variable name.
    check_inside : bool
        To check the elements of the object, rather than the object itself.

    Returns
    -------
    var_value : type(var_name)
        Value of variable `var_name`.

    """

    check = False

    if check_inside:
        if isinstance(var_value, Iterable):
            if all(isinstance(elem, valid_types) for elem in var_value):
                check = True
        else:
            custom_error(
                var_name,
                var_value,
                TypeError,
                f"`Variable {var_name}` should be iterable.",
            )

    else:
        if isinstance(var_value, valid_types):
            check = True

    if check:
        return var_value
    else:
        custom_error(
            var_name,
            var_value,
            TypeError,
            f"Valid `{var_name}` types are: {valid_types}.",
        )


def check_subclass(myobject: Any,
                   valid_class: Union[type, tuple[type, ...]],
                   myobject_name: str,
                   check_inside: bool = False
                   ) -> Any:
    """Check if an object is a subclass of given class."""

    check = False

    if check_inside:
        if all(issubclass(elem, valid_class) for elem in myobject):
            check = True
    else:
        if issubclass(myobject, valid_class):
            check = True

    if check:
        return myobject
    else:
        custom_error(
            myobject_name,
            myobject,
            TypeError,
            f"Valid `{myobject_name}` types are: {valid_class}.",
        )


def check_bounds(x: Union[float, np.ndarray],
                 xmin: float,
                 xmax: float,
                 xname: str
                 ) -> Union[float, np.ndarray, None]:
    """Check if a numerical value is between given bounds.

    Example:
    -------
    >>> check_bounds(1.0, 0.1, 2.0, 'x') -> 1.0
    >>> check_bounds(-1.0, 0.1, 2.0, 'x') -> RangeError

    Parameters
    ----------
    x : float | ndarray
        Variable.
    xmin : float
        Lower bound.
    xmax : float
        Upper bound.
    xname : str
        Variable name.

    Returns
    -------
    x : float | ndarray
        Variable.

    """
    if isinstance(x, numbers.Number) and (x >= xmin) and (x <= xmax):
        return x
    elif isinstance(x, np.ndarray) and \
            np.all(np.logical_and.reduce((x >= xmin, x <= xmax))):
        return x
    else:
        check_type(x, (numbers.Number, np.ndarray), xname)
        custom_error(
            xname, x, RangeError, f"Valid `{xname}` range is [{xmin}, {xmax}]."
        )


def check_in_set(var_value: Any,
                 valid_set: set,
                 var_name: str = "#"
                 ) -> Any:
    """Check if a variable or its elements belong to a set.

    Example:
    -------
    >>> check_value_set('A', {'B','A'}) -> 'A'
    """

    # Check valid_set is a set
    check_type(valid_set, set, "valid_set")

    # Convert to iterable if necessary
    if not (isinstance(var_value, (list, set, tuple))):
        var_set = set([var_value])
    else:
        var_set = set(var_value)

    # Compare sets
    if var_set <= valid_set:
        return var_value
    else:
        diff_set = var_set - valid_set
        custom_error(
            var_name,
            var_value,
            ValueError,
            f"The valid set is: {valid_set}. "
            f"The following items do not belong to the valid set: {diff_set}.",
        )


def check_shapes(a: list, b: list) -> Union[tuple[int, ...], None]:
    """Check shape homogeneity between objects in lists `a` and `b`.

    Rules:
    - All objects in `a` must have the same shape, i.e., either all floats
    or all arrays with same shape.
    - Objects in `b` that are arrays, must have identical shape to the
    objects in `a`.

    Parameters
    ----------
    a : list
        List of objects which must have the same shape.
    b : list
        List of objects which, if arrays, must have identical shape to the
        objects in `a`.

    Returns
    -------
    Union[tuple[int, ...], None]
        Common shape of `a` or None.
    """

    check_a = True
    check_b = True
    shape = None
    shapes_a = [elem.shape for elem in a if isinstance(elem, np.ndarray)]
    shapes_b = [elem.shape for elem in b if isinstance(elem, np.ndarray)]
    if shapes_a:
        if len(shapes_a) != len(a) or len(set(shapes_a)) != 1:
            check_a = False
        else:
            shape = shapes_a[0]
    if shapes_b:
        if len(set(shapes_a + shapes_b)) != 1:
            check_b = False
    if not (check_a and check_b):
        raise ShapeError("Input parameters have inconsistent shapes.")
    return shape


def check_valid_range(r: tuple[float, float], name: str) -> None:
    "Check is a given input range is a valid range."
    if not (len(r) == 2 and r[1] > r[0]):
        raise RangeError(f"`{name}` is invalid: {r}")

# %% Special functions


def add_dicts(d1: dict[Any, Union[int, float]],
              d2: dict[Any, Union[int, float]],
              new: bool = False
              ) -> dict[Any, Union[int, float]]:
    """Adds two dictionaries by summing the values for the same key.

    Parameters
    ----------
    d1 : dict[Any, int  |  float]
        first dictionary
    d2 : dict[Any, int  |  float]
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
        dout[key] = dout.get(key, 0) + value
    return dout


class vectorize(np.vectorize):
    "Vectorize decorator for instance methods."

    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

# %% Unit functions


def convert_check_temperature(T: FloatOrArrayLike,
                              Tunit: Literal['C', 'K'],
                              Tmin: FloatOrArray = 0.,
                              Tmax: FloatOrArray = np.inf
                              ) -> FloatOrArray:
    """Convert temperature input to K and check range.

    Parameters
    ----------
    T : FloatOrArrayLike
        Temperature
    Tunit : Literal['C', 'K']
        Temperature unit.
    Tmin : FloatOrArray
        Lower temperature bound.
        Unit = K.
    Tmax : FloatOrArray
        Upper temperature bound.
        Unit = K.

    Returns
    -------
    FloatOrArray
        Temperature in K.
    """

    if isinstance(T, list):
        T = np.array(T, dtype=np.float64)
    if Tunit == 'K':
        TK = T
    elif Tunit == 'C':
        TK = T + 273.15
    else:
        raise ValueError("Invalid `Tunit` input.")
    if np.any(TK < 0):
        raise RangeError("`T` must be > 0 K.")
    if np.any(TK < Tmin) or np.any(TK > Tmax):
        print("Warning: `T` input is outside validity range [Tmin, Tmax].")
    return TK
