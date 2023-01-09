from typing import Any, Union
from collections.abc import Iterable
import numbers

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
    error_message = f"`{var_value}` is not a valid value for `{var_name}`."
    # sys.tracebacklimit = 1
    if message != "":
        error_message += " " + message
    raise kind(error_message)
    # sys.exit(0)


def check_type(var_value: Any,
               valid_types: Any,
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
                   valid_class: type,
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


def check_bounds(x: Union[int, float],
                 xmin: Union[int, float],
                 xmax: Union[int, float],
                 xname: str
                 ) -> Union[int, float, None]:
    """Check if a numerical value is between given bounds.

    Example:
    -------
    >>> check_bounds(1.0, 0.1, 2.0, 'x') -> 1.0
    >>> check_bounds(-1.0, 0.1, 2.0, 'x') -> ValueError

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
    x : float
        Variable.

    """

    x = check_type(x, numbers.Number, xname)

    if x >= xmin and x <= xmax:
        return x
    else:
        custom_error(
            xname, x, ValueError, f"Valid `{xname}` range is [{xmin}, {xmax}]."
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
            f"""The following items do not belong to the valid set: {diff_set}.
            The valid set is: {valid_set}.""",
        )

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
        d1 will be modified (`d1 <- d1 + d2`).

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
