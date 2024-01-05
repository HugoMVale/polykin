# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from polykin.properties.equations.base import PropertyEquationT
from polykin.types import FloatOrArray


class KineticCoefficientT(PropertyEquationT):

    _shape: Optional[tuple]

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        """Shape of underlying parameter array."""
        return self._shape


class KineticCoefficientCLD(ABC):
    """_Abstract_ chain-length dependent (CLD) coefficient."""

    name: str

    @abstractmethod
    def __call__(self, T, i, *args) -> FloatOrArray:
        pass

    @staticmethod
    @abstractmethod
    def equation(T, i, *args) -> FloatOrArray:
        pass
