# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from dataclasses import dataclass

from polykin.utils.types import FloatVector

__all__ = [
    'RootResult',
    'VectorRootResult'
]


@dataclass
class RootResult():
    """Dataclass with root solution results.

    Attributes
    ----------
    success: bool
        If `True`, the root was found.
    message: str
        Description of the exit status.
    nfeval: int
        Number of function evaluations.
    niter: int
        Number of iterations.
    x: float
        Root value.
    f: float
        Function (residual) value at root.
    """
    success: bool
    message: str
    nfeval: int
    niter: int
    x: float
    f: float


@dataclass
class VectorRootResult():
    """Dataclass with vector root solution results.

    Attributes
    ----------
    success: bool
        If `True`, the root was found.
    message: str
        Description of the exit status.
    nfeval: int
        Number of function evaluations.
    niter: int
        Number of iterations.
    x: FloatVector
        Root value.
    f: FloatVector
        Function (residual) value at root.
    """
    success: bool
    message: str
    nfeval: int
    niter: int
    x: FloatVector
    f: FloatVector
