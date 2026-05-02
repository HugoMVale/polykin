# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

from dataclasses import dataclass

from polykin.utils.tools import colored_bool
from polykin.utils.typing import FloatMatrix, FloatVector

__all__ = [
    "OptimumResult",
    "VectorOptimumResult",
]


@dataclass(frozen=True, slots=True)
class OptimumResult:
    """Dataclass with the results of the optimization.

    Attributes
    ----------
    method: str
        Method used to find the optimum.
    success: bool
        If `True`, the optimum was found.
    message: str
        Description of the exit status.
    nfeval: int
        Number of function evaluations.
    niter: int
        Number of iterations.
    x: float
        Optimum value.
    f: float
        Function value at the optimum.
    df: float | None
        Derivative at the optimum (if available).
    """

    method: str
    success: bool
    message: str
    nfeval: int
    niter: int
    x: float
    f: float
    df: float | None = None

    def __repr__(self) -> str:
        """Return a string representation of the optimum result."""
        return (
            f" method: {self.method}\n"
            f"success: {colored_bool(self.success)}\n"
            f"message: {self.message}\n"
            f" nfeval: {self.nfeval}\n"
            f"  niter: {self.niter}\n"
            f"      x: {self.x}\n"
            f"      f: {self.f}\n"
            f"     df: {self.df}"
        )


@dataclass(frozen=True, slots=True)
class VectorOptimumResult:
    """Dataclass with the results of the optimization.

    Attributes
    ----------
    method: str
        Method used to find the optimum.
    success: bool
        If `True`, the optimum was found.
    message: str
        Description of the exit status.
    nfeval: int
        Number of function evaluations.
    ngeval: int | None
        Number of user-supplied gradient evaluations.
    nheval: int | None
        Number of user-supplied Hessian evaluations.
    niter: int
        Number of iterations.
    x: FloatVector
        Optimum value.
    f: float
        Function value at the optimum.
    g: FloatVector | None
        Gradient at the optimum.
    H: FloatMatrix | None
        Last evaluated or estimated Hessian matrix. For BFGS-based updates, this may
        represent the dense Hessian, the inverse Hessian ($H^{-1}$), or the Cholesky
        factor $L$ (where $H = LL^T$), depending on the `method` used.
    """

    method: str
    success: bool
    message: str
    nfeval: int
    ngeval: int | None
    nheval: int | None
    niter: int
    x: FloatVector
    f: float
    g: FloatVector | None = None
    H: FloatMatrix | None = None

    def __repr__(self) -> str:
        """Return a string representation of the optimum result."""
        return (
            f" method: {self.method}\n"
            f"success: {colored_bool(self.success)}\n"
            f"message: {self.message}\n"
            f" nfeval: {self.nfeval}\n"
            f" ngeval: {self.ngeval}\n"
            f" nheval: {self.nheval}\n"
            f"  niter: {self.niter}\n"
            f"      x: {self.x}\n"
            f"      f: {self.f}\n"
            f"      g: {self.g}\n"
            f"      H: {self.H}"
        )
