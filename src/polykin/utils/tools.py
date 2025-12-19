# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""
Input validation, unit conversion, and formatting utilities for PolyKin.

This module provides a collection of helper functions used throughout
PolyKin to validate input types, ranges, and shapes; perform basic unit
conversions for thermodynamic quantities; and generate readable string
representations for numerical objects. The utilities are designed to
support both scalar and NumPy array inputs and to raise clear,
domain-specific exceptions when invalid data is encountered.
"""

from collections.abc import Iterable
from numbers import Real
from typing import Any, Literal
from warnings import warn

import numpy as np

from .exceptions import RangeError, ShapeError
from .types import FloatMatrix, FloatOrArray, FloatOrArrayLike


def custom_error(
    var_name: str,
    var_value: Any,
    kind: type,
    message: str = "",
) -> None:
    """
    Raise a custom error message.

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
    error_message = f"`{var_name} = {var_value}` is not a valid input."
    # sys.tracebacklimit = 1
    if message != "":
        error_message += " " + message
    # print(f"{kind}: {error_message}")
    # sys.exit(1)
    raise kind(error_message)


def check_type(
    var_value: Any,
    valid_types: type | tuple[type, ...],
    var_name: str,
    check_inside: bool = False,
) -> Any:
    """Check if a variable is of a given type.

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
            f"Variable `{var_name}` is of type {type(var_value)}. Valid `{var_name}` types are: {valid_types}.",  # noqa: E501
        )


def check_subclass(
    myobject: Any,
    valid_class: type | tuple[type, ...],
    myobject_name: str,
    check_inside: bool = False,
) -> Any:
    """Check if an object is a subclass of given class.

    Parameters
    ----------
    myobject : Any
        Object to check.
    valid_class : type | tuple[type, ...]
        Valid class or tuple of classes.
    myobject_name : str
        Name of the object.
    check_inside : bool
        To check the elements of the object, rather than the object itself.

    Returns
    -------
        myobject : Any
            Object.
    """
    check = False

    if check_inside:
        if not isinstance(myobject, Iterable):
            custom_error(
                myobject_name,
                myobject,
                TypeError,
                f"`{myobject_name}` should be iterable.",
            )
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


def check_bounds(
    x: float | np.ndarray | Iterable,
    xmin: float,
    xmax: float,
    xname: str,
) -> float | np.ndarray | Iterable | None:
    """Check if a numerical value is between given bounds.

    Parameters
    ----------
    x : float | ndarray | Iterable
        Variable whose bounds are to be checked.
    xmin : float
        Lower bound.
    xmax : float
        Upper bound.
    xname : str
        Variable name.

    Returns
    -------
    x : float | ndarray | Iterable
        Variable.

    Examples
    --------
    >>> check_bounds(1.0, 0.1, 2.0, 'x')
    1.0
    >>> check_bounds(-1.0, 0.1, 2.0, 'x') # RangeError
    """
    if isinstance(x, (int, float, Real)) and ((x >= xmin) and (x <= xmax)):
        return x
    elif isinstance(x, np.ndarray) and np.all(
        np.logical_and.reduce((x >= xmin, x <= xmax))
    ):
        return x
    elif isinstance(x, Iterable) and all(xi >= xmin and xi <= xmax for xi in x):
        return x
    else:
        check_type(x, (int, float, Real, np.ndarray, Iterable), xname)
        custom_error(
            xname, x, RangeError, f"Valid `{xname}` range is [{xmin}, {xmax}]."
        )


def check_in_set(
    var_value: Any,
    valid_set: set,
    var_name: str = "#",
) -> Any:
    """Check if a variable or its elements belong to a set.

    Notes
    -----
    Elements must be hashable.

    Parameters
    ----------
    var_value : type(var_name)
        Value of variable `var_name`.
    valid_set : set
        Valid set of values.
    var_name : str
        Variable name.

    Returns
    -------
    type(var_name)
        Value of variable `var_name`.

    Examples
    --------
    >>> check_value_set('A', {'B','A'})
    'A'
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


def check_shapes(
    a: list[float | np.ndarray],
    b: list[float | np.ndarray] | None = None,
) -> tuple[int, ...] | None:
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
    b : list | None
        List of objects which, if arrays, must have identical shape to the
        objects in `a`.

    Returns
    -------
    tuple[int, ...] | None
        Common shape of `a` or None.
    """
    if b is None:
        b = []

    shapes_a = [
        elem.shape for elem in a if isinstance(elem, np.ndarray) and elem.shape != ()
    ]
    shapes_b = [
        elem.shape for elem in b if isinstance(elem, np.ndarray) and elem.shape != ()
    ]

    check_a = True
    shape = None
    if shapes_a:
        check_a = len(shapes_a) == len(a) and len(set(shapes_a)) == 1
        shape = shapes_a[0]

    check_b = True
    if shapes_b:
        if not shapes_a:
            check_b = False
        else:
            if len(set(shapes_a + shapes_b)) != 1:
                check_b = False

    if not check_a or not check_b:
        raise ShapeError("Input parameters have inconsistent shapes.")

    return shape


def check_valid_range(
    r: tuple[float, float],
    xmin: float,
    xmax: float,
    name: str,
) -> tuple[float, float] | None:
    """Check if a given input range is a valid range.

    Parameters
    ----------
    r : tuple[float, float]
        Range.
    xmin : float
        Lower bound.
    xmax : float
        Upper bound.
    name : str
        Variable name.

    Returns
    -------
    tuple[float, float] | None
        Range.
    """
    if not (
        isinstance(r, tuple)
        and len(r) == 2
        and isinstance(r[0], Real)
        and isinstance(r[1], Real)
        and r[1] > r[0]
    ):
        raise RangeError(f"`{name}` is invalid: {r}")
    check_bounds(r, xmin, xmax, name)
    return r


def check_range_warn(
    x: float,
    xmin: float,
    xmax: float,
    xname: str,
) -> None:
    """Check if a given input is in a valid range and warn if not.

    Parameters
    ----------
    x : float
        Variable.
    xmin : float
        Lower bound.
    xmax : float
        Upper bound.
    xname : str
        Variable name.

    Returns
    -------
    None
    """
    if x < xmin or x > xmax:
        # Intentionally made without f-strings, so it works with Numba
        print(xname + "=", x, "is outside the valid range [", xmin, ",", xmax, "]")


def convert_check_temperature(
    T: FloatOrArrayLike,
    Tunit: Literal["C", "K"],
    Trange: tuple[FloatOrArray, FloatOrArray] = (0.0, np.inf),
) -> FloatOrArray:
    r"""Convert temperature input to 'K' and check range.

    Parameters
    ----------
    T : FloatOrArrayLike
        Temperature
    Tunit : Literal['C', 'K']
        Temperature unit.
    Trange : FloatOrArray
        Temperature range.
        Unit = K.

    Returns
    -------
    FloatOrArray
        Temperature in K.
    """
    if isinstance(T, (list, tuple)):
        T = np.array(T, dtype=np.float64)

    if Tunit == "K":
        TK = T
    elif Tunit == "C":
        TK = T + 273.15
    else:
        raise ValueError("Invalid `Tunit` input.")

    if np.any(TK < 0.0):
        raise RangeError("`T` must be > 0 K.")
    if np.any(TK < Trange[0]) or np.any(TK > Trange[1]):
        warn(f"`T` input is outside validity range {Trange}.")

    return TK


def convert_check_pressure(
    P: FloatOrArrayLike,
    Punit: Literal["bar", "MPa", "Pa"],
    Prange: tuple[FloatOrArray, FloatOrArray] = (0.0, np.inf),
) -> FloatOrArray:
    """Convert pressure input to 'Pa' and check range.

    Parameters
    ----------
    P : FloatOrArrayLike
        Pressure
    Punit : Literal['bar', 'MPa', Pa']
        Pressure unit.
    Prange : FloatOrArray
        Pressure range.
        Unit = Pa.

    Returns
    -------
    FloatOrArray
        Pressure in Pa.
    """
    if isinstance(P, (list, tuple)):
        P = np.array(P, dtype=np.float64)

    if Punit == "Pa":
        f = 1
    elif Punit == "bar":
        f = 1e5
    elif Punit == "MPa":
        f = 1e6
    else:
        raise ValueError("Invalid `Punit` input.")
    Pa = P * f

    if np.any(Pa < 0.0):
        raise RangeError("`P` must be > 0 Pa.")
    if np.any(Pa < Prange[0]) or np.any(Pa > Prange[1]):
        warn(f"`P` input is outside validity range {Prange}.")

    return Pa


def custom_repr(
    obj,
    attr_names: list[str] | tuple[str, ...],
    nspaces: int = 3,
) -> str:
    """Generate custom repr string.

    Parameters
    ----------
    obj : Any
        Object.
    attr_names : list[str]
        Class atributes.
    nspaces : int
        Number of white spaces after longest attribute name.

    Returns
    -------
    str
        Formated repr string.
    """
    if not attr_names:
        return ""

    items = []
    nspaces += max([len(attr) for attr in attr_names])
    for attr_name in attr_names:
        attr_str = str(getattr(obj, attr_name))
        rows = attr_str.split("\n")
        if len(rows) == 1:
            item = f"{attr_name}:" + " " * (nspaces - len(attr_name)) + attr_str
        else:
            item = "\n  ".join([attr_name + ":"] + rows)
        items.append(item)

    return "\n".join(items)


def pprint_matrix(
    matrix: FloatMatrix,
    format_specifier="{:.2e}",
    nspaces: int = 0,
) -> str:
    """Pretty print a matrix (2D array).

    Parameters
    ----------
    matrix : FloatMatrix
        Matrix to print.
    format_specifier : str
        Format specifier, e.g "{:.2f}".
    nspaces : int
        Number of white spaces placed before 2nd row and following.

    Returns
    -------
    str
        Formated matrix string.
    """
    nrows = matrix.shape[0]
    result = ""
    for i, row in enumerate(matrix):
        line = "[[" if i == 0 else (" " * nspaces + " [")
        line += " ".join(format_specifier.format(element) for element in row)
        line += "]]\n" if i == nrows - 1 else "]\n"
        result += line
    return result


def colored_bool(value: bool) -> str:
    """Color boolean as green if `True`, red if `False`.

    Notes
    -----
    ANSI escape codes may not render correctly on Windows consoles or in logs

    Parameters
    ----------
    value : bool
        Boolean.

    Returns
    -------
    str
        Colored boolean.
    """
    green, red, reset = "\033[92m", "\033[91m", "\033[0m"
    color = green if value else red
    return f"{color}{value}{reset}"
