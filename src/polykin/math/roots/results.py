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

    def __repr__(self) -> str:
        green, red, reset = "\033[92m", "\033[91m", "\033[0m"
        color = green if self.success else red
        return (f"success: {color}{self.success}{reset}\n"
                f"message: {self.message}\n"
                f" nfeval: {self.nfeval}\n"
                f"  niter: {self.niter}\n"
                f"      x: {self.x}\n"
                f"      f: {self.f}")


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

    def __repr__(self) -> str:
        green, red, reset = "\033[92m", "\033[91m", "\033[0m"
        color = green if self.success else red
        return (f"success: {color}{self.success}{reset}\n"
                f"message: {self.message}\n"
                f" nfeval: {self.nfeval}\n"
                f"  niter: {self.niter}\n"
                f"      x: {self.x}\n"
                f"      f: {self.f}")
