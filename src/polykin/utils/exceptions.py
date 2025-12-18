# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""
Custom exceptions and warnings used throughout PolyKin.

This module defines a consistent set of error and warning types for common
failure modes in polymerization kinetics workflows, including convergence
issues, fitting failures, ODE solver problems, root-finding errors, and
invalid ranges or shapes. These classes are intended to improve clarity,
error handling, and diagnostics across the library.
"""


class ConvergenceError(ValueError):
    """Raised when an iterative algorithm fails to converge to a solution."""


class ConvergenceWarning(Warning):
    """
    Raised when an iterative algorithm is approaching non-convergence,
    indicating potential instability or slow convergence.
    """


class FitError(ValueError):
    """Raised when a fitting procedure fails to find a valid solution."""


class FitWarning(Warning):
    """
    Raised when a fitting procedure produces results that may be unreliable
    or near failure.
    """


class ODESolverError(ValueError):
    """
    Raised when an ordinary differential equation (ODE) solver fails to
    produce a solution.
    """


class ODESolverWarning(Warning):
    """
    Raised when an ODE solver produces results that may be unreliable
    or indicate potential failure.
    """


class RangeError(ValueError):
    """Raised when a value lies outside the allowed or expected range."""


class RangeWarning(Warning):
    """
    Raised when a value is near or slightly outside the recommended range,
    but not critically so.
    """


class RootSolverError(ValueError):
    """Raised when a root-finding solver fails to locate a valid root."""


class RootSolverWarning(Warning):
    """
    Raised when a root-finding solver encounters potential numerical issues
    or uncertain results.
    """


class ShapeError(ValueError):
    """Raised when an array or tensor has an invalid or unexpected shape."""
