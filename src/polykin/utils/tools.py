# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from numbers import Number
from typing import Any, Iterable, Literal, Union

import numpy as np

from .exceptions import RangeError, ShapeError
from .types import FloatOrArray, FloatOrArrayLike, FloatMatrix

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
            f"Variable `{var_name}` if of type {type(var_value)}. Valid `{var_name}` types are: {valid_types}."  # noqa: E501
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


def check_bounds(x: Union[float, np.ndarray, Iterable],
                 xmin: float,
                 xmax: float,
                 xname: str
                 ) -> Union[float, np.ndarray, Iterable, None]:
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
    if isinstance(x, (float, Number)) and ((x >= xmin) and (x <= xmax)):
        return x
    elif isinstance(x, np.ndarray) and \
            np.all(np.logical_and.reduce((x >= xmin, x <= xmax))):
        return x
    elif isinstance(x, Iterable) and \
            all((xi >= xmin and xi <= xmax for xi in x)):
        return x
    else:
        check_type(x, (float, Number, np.ndarray, Iterable), xname)
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


def check_shapes(a: list[Union[float, np.ndarray]],
                 b: list[Union[float, np.ndarray]] = []
                 ) -> Union[tuple[int, ...], None]:
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

    shapes_a = [elem.shape for elem in a if
                isinstance(elem, np.ndarray) and elem.shape != ()]
    shapes_b = [elem.shape for elem in b if
                isinstance(elem, np.ndarray) and elem.shape != ()]

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


def check_valid_range(r: tuple[float, float],
                      xmin: float,
                      xmax: float,
                      name: str
                      ) -> Union[tuple[float, float], None]:
    "Check if a given input range is a valid range."
    check_type(r, tuple, name)
    if not (len(r) == 2 and r[1] > r[0]):
        raise RangeError(f"`{name}` is invalid: {r}")
    check_bounds(r, xmin, xmax, name)
    return r


# %% Unit functions


def convert_check_temperature(
        T: FloatOrArrayLike,
        Tunit: Literal['C', 'K'],
        Trange: tuple[FloatOrArray, FloatOrArray] = (0., np.inf)) -> FloatOrArray:
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

    if Tunit == 'K':
        TK = T
    elif Tunit == 'C':
        TK = T + 273.15
    else:
        raise ValueError("Invalid `Tunit` input.")

    if np.any(TK < 0):
        raise RangeError("`T` must be > 0 K.")
    if np.any(TK < Trange[0]) or np.any(TK > Trange[1]):
        print(f"Warning: `T` input is outside validity range {Trange}.")

    return TK


def convert_check_pressure(
        P: FloatOrArrayLike,
        Punit: Literal['bar', 'MPa', 'Pa'],
        Prange: tuple[FloatOrArray, FloatOrArray] = (0., np.inf)) -> FloatOrArray:
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

    if Punit == 'Pa':
        f = 1
    elif Punit == 'bar':
        f = 1e5
    elif Punit == 'MPa':
        f = 1e6
    else:
        raise ValueError("Invalid `Punit` input.")
    Pa = P*f

    if np.any(Pa < 0):
        raise RangeError("`P` must be > 0 Pa.")
    if np.any(Pa < Prange[0]) or np.any(Pa > Prange[1]):
        print(f"Warning: `P` input is outside validity range {Prange}.")

    return Pa


def custom_repr(obj,
                attr_names: Union[list[str], tuple[str, ...]],
                nspaces: int = 3
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
    items = []
    nspaces += max([len(attr) for attr in attr_names])
    for attr_name in attr_names:
        attr_str = str(getattr(obj, attr_name))
        rows = attr_str.split("\n")
        if len(rows) == 1:
            item = f"{attr_name}:" + " "*(nspaces - len(attr_name)) + attr_str
        else:
            item = "\n  ".join([attr_name + ":"] + rows)
        items.append(item)
    return "\n".join(items)


def pprint_matrix(matrix: FloatMatrix,
                  format_specifier="{:.2e}",
                  nspaces: int = 0) -> str:
    """Pretty print a matrix.

    Parameters
    ----------
    matrix : FloatMatrix
        Matrix to print.
    format_specifier : str
        Format specifier, e.g "{:.2f}".
    nspaces : int
        Number of white spaces placed before 2nd row and following.
    """
    nrows = matrix.shape[0]
    result = ""
    for i, row in enumerate(matrix):
        line = "[[" if i == 0 else (" "*nspaces + " [")
        line += " ".join(format_specifier.format(element) for element in row)
        line += "]]\n" if i == nrows-1 else "]\n"
        result += line
    return result
