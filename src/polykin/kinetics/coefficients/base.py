# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from __future__ import annotations

from abc import ABC, abstractmethod

from polykin.properties.equations.base import PropertyEquationT
from polykin.utils.types import FloatArray


class KineticCoefficientT(PropertyEquationT):
    """Abstract temperature-dependent kinetic coefficient."""

    _shape: tuple | None

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Shape of underlying parameter array."""
        return self._shape


class KineticCoefficientCLD(ABC):
    """Abstract chain-length dependent (CLD) coefficient."""

    name: str

    @abstractmethod
    def __call__(self, T, i, *args) -> float | FloatArray:
        """Evaluate the coefficient."""

    @staticmethod
    @abstractmethod
    def equation(T, i, *args) -> float | FloatArray:
        """Equation for the coefficient."""
