# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

from dataclasses import dataclass

from polykin.utils.tools import colored_bool

__all__ = ["OptimumResult"]


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
    """

    method: str
    success: bool
    message: str
    nfeval: int
    niter: int
    x: float
    f: float

    def __repr__(self) -> str:
        """Return a string representation of the optimum result."""
        return (
            f" method: {self.method}\n"
            f"success: {colored_bool(self.success)}\n"
            f"message: {self.message}\n"
            f" nfeval: {self.nfeval}\n"
            f"  niter: {self.niter}\n"
            f"      x: {self.x}\n"
            f"      f: {self.f}"
        )
