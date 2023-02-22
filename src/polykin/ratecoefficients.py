# %% Rate coefficients

from polykin.utils import FloatOrArray, FloatOrArrayLike, check_bounds
from polykin.base import Base
from scipy.constants import Planck, Boltzmann, gas_constant

import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod


class KineticCoefficient(Base, ABC):

    @abstractmethod
    def __init__(self, *args):
        pass

    def __call__(self,
                 T: FloatOrArrayLike,
                 kelvin: bool = False
                 ) -> FloatOrArray:
        r"""Evaluate rate coefficient at desired temperature.

        Parameters
        ----------
        T : float | list[float] | ndarray
            Temperature (default °C).
        kelvin : bool, optional
            Switch temperature unit between °C (if `False`) and K (if `True`).

        Returns
        -------
        float | ndarray
            Rate coefficient, k(T).
        """
        if isinstance(T, list):
            T = np.asarray(T, dtype=np.float64)
        if kelvin:
            TK = T
        else:
            TK = T + 273.15
        if np.any(TK < 0):  # adds 5 us
            raise ValueError("Invalid `T` input.")
        return self.eval(TK)

    @abstractmethod
    def eval(self, TK: FloatOrArray) -> FloatOrArray:
        pass


class Arrhenius(KineticCoefficient):

    def __init__(self, k0: FloatOrArrayLike,
                 EaR: FloatOrArrayLike,
                 T0: FloatOrArrayLike = np.inf,
                 name: str = ''):

        # convert lists to arrays
        if isinstance(k0, list):
            k0 = np.asarray(k0, dtype=np.float64)
        if isinstance(EaR, list):
            EaR = np.asarray(EaR, dtype=np.float64)
        if isinstance(T0, list):
            T0 = np.asarray(T0, dtype=np.float64)

        # check shape consistency
        if isinstance(k0, ndarray):
            if not isinstance(EaR, ndarray) or not (k0.shape == EaR.shape):
                raise ValueError("Shapes of `k0` and `EaR` are inconsistent.")
            if isinstance(T0, ndarray) and not (k0.shape == T0.shape):
                raise ValueError("Shapes of `k0` and `T0` are inconsistent.")

        # check bounds
        check_bounds(k0, 0, np.inf, 'k0')
        check_bounds(EaR, 0, np.inf, 'EaR')
        check_bounds(T0, -273.15, np.inf, 'T0')

        self.k0 = k0
        self.EaR = EaR
        self.T0 = T0 + 273.15
        self.name = name

    def eval(self, TK):
        return self.k0*np.exp(-self.EaR*(1/TK - 1/self.T0))


class Eyring(KineticCoefficient):

    def __init__(self,
                 DGa: FloatOrArrayLike,
                 kappa: FloatOrArrayLike,
                 name: str = ''):

        # convert lists to arrays
        if isinstance(DGa, list):
            DGa = np.asarray(DGa, dtype=np.float64)
        if isinstance(kappa, list):
            kappa = np.asarray(kappa, dtype=np.float64)

        if isinstance(kappa, ndarray):
            if not isinstance(DGa, ndarray) or not (kappa.shape == DGa.shape):
                raise ValueError(
                    "Shapes of `DGa` and `kappa` are inconsistent.")

        # check bounds
        check_bounds(DGa, 0, np.inf, 'DGa')
        check_bounds(kappa, 0, 1, 'kappa')

        self.DGa = DGa
        self.kappa = kappa
        self.name = name

    def eval(self, TK):
        return self.kappa*Boltzmann*TK/Planck*np.exp(-self.DGa/(gas_constant*TK))
