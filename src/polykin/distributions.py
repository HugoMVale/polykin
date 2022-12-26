# %%

from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):
    """Abstract class for all chain-length distributions."""

    def __init__(self, DPn: int = 100, molar_mass: float = 100.0):
        self.DPn = DPn
        self.molar_mass = molar_mass

    def __call__(self, x, dist: str = "mass", unit_x: str = "chain_length"):

        # Select unit_x
        if unit_x == "chain_length":
            pass
        elif unit_x == "molar_mass":
            x = self._list_to_array(x) / self.DPn
        else:
            raise ValueError

        # Select distribution
        if dist == "number":
            w = self.dist_number(x)
        elif dist == "mass":
            w = self.dist_mass(x)
        elif dist == "gpc":
            w = self.dist_gpc(x)
        else:
            raise ValueError

        return w

    def dist_number(self, length):
        return self._pmf(self._list_to_array(length))

    def dist_mass(self, length):
        return self.dist_number(length) * length

    def dist_gpc(self, length):
        return self.dist_number(length) * length**2

    @abstractmethod
    def _pmf(self, length):
        return 0

    @staticmethod
    def _list_to_array(length):
        if isinstance(length, list):
            length = np.asarray(length)
        return length


class Flory(Distribution):
    """Flory-Schulz (aka most-probable) chain-length distribution.
    https://en.wikipedia.org/wiki/Flory%E2%80%93Schulz_distribution
    """

    def _pmf(self, i):
        a = 2.0 / (self.DPn + 1)
        w = a**2 * i * (1 - a) ** (i - 1)
        return w


class Poisson(Distribution):
    pass
