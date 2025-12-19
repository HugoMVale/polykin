# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from dataclasses import dataclass

from polykin.utils.tools import colored_bool
from polykin.utils.types import FloatMatrix, FloatVector

__all__ = [
    "RootResult",
    "VectorRootResult",
]


@dataclass
class RootResult:
    """Dataclass with scalar root solution results.

    Attributes
    ----------
    method: str
        Method used to find the root.
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

    method: str
    success: bool
    message: str
    nfeval: int
    niter: int
    x: float
    f: float

    def __repr__(self) -> str:
        """Return a string representation of the root result."""
        return (
            f" method: {self.method}\n"
            f"success: {colored_bool(self.success)}\n"
            f"message: {self.message}\n"
            f" nfeval: {self.nfeval}\n"
            f"  niter: {self.niter}\n"
            f"      x: {self.x}\n"
            f"      f: {self.f}"
        )


@dataclass
class VectorRootResult:
    """Dataclass with vector root solution results.

    Attributes
    ----------
    method: str
        Method used to find the root.
    success: bool
        If `True`, the root was found.
    message: str
        Description of the exit status.
    nfeval: int
        Number of function evaluations.
    njeval: int | None
        Number of Jacobian evaluations.
    niter: int
        Number of iterations.
    x: FloatVector
        Root value.
    f: FloatVector
        Function (residual) value at root.
    jac: FloatMatrix
        Last Jacobian value evaluated or estimated.
    """

    method: str
    success: bool
    message: str
    nfeval: int
    njeval: int | None
    niter: int
    x: FloatVector
    f: FloatVector
    jac: FloatMatrix | None

    def __repr__(self) -> str:
        """Return a string representation of the vector root result."""
        return (
            f" method: {self.method}\n"
            f"success: {colored_bool(self.success)}\n"
            f"message: {self.message}\n"
            f" nfeval: {self.nfeval}\n"
            f" njeval: {self.njeval}\n"
            f"  niter: {self.niter}\n"
            f"      x: {self.x}\n"
            f"      f: {self.f}\n"
            f"    jac: {self.jac}"
        )
