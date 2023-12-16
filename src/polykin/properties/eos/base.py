# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector, FloatSquareMatrix

import numpy as np
from numpy import exp, sqrt, log
from scipy.constants import R
from abc import ABC, abstractmethod
from typing import Optional, Literal

__all__ = ['EoS',
           'Ideal']

# Standard pressure


class EoS(ABC):

    P0 = 1e5  # Pa

    def V(self,
          T: float,
          P: float,
          y: FloatVector) -> FloatVector:
        r"""Calculate the molar volumes of the coexisting phases a fluid.

        $$  V = \frac{Z R T}{P}

        where $V$ is the molar volue, $Z(T,P,y)$ is the compressibility factor,
        $T$ is the temperature, $P$ is the pressure, and $y$ is the mole
        fraction vector.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        P : float
            Pressure. Unit = Pa.
        y : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector
            Molar volume of the gas and/or liquid phases. Unit = m³/mol.
        """
        return self.Z(T, P, y)*R*T/P

    @abstractmethod
    def Z(self, T: float, P: float, y: FloatVector) -> FloatVector:
        r"""Calculate the compressibility factors of the coexisting phases a
        fluid.

        $$  Z = \frac{P V}{R T}

        where $Z$ is the compressibility factor, $P$ is the pressure, $T$ is
        the temperature, $V(T,P,Z)$ is the molar volume, and $y$ is the
        mole fraction vector.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        P : float
            Pressure. Unit = Pa.
        y : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector
            Compressibility factor of the gas and/or liquid phases.
        """
        pass

    @abstractmethod
    def P(self, T: float, V: float, y: FloatVector) -> float:
        r"""Calculate the equilibrium pressure of a fluid.

        $$  Z = \frac{P V(T,P,y)}{R T}

        where $Z$ is the compressibility factor, $T$ is the temperature,
        $P$ is the pressure, and $y$ is the vector of mole fractions.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        P : float
            Pressure. Unit = Pa.
        y : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector
            Compressibility factor of the gas and/or liquid phases.
        """
        pass

    @abstractmethod
    def phi(self, T: float, P: float, y: FloatVector) -> FloatVector:
        r"""Calculate the fugacity coefficients of all components in the
        mixture.

        $$ \phi_i = \frac{f_i^V}{y_i P} $$

        where $\phi_i$ is the fugacity coefficient, $f_i^V$ is the fugacity in
        the vapor phase, $P$ is the pressure, and $y_i$ is the mole fraction in
        the vapor phase.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        P : float
            Pressure. Unit = Pa.
        y : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector
            Fugacity coefficients of all components.
        """
        pass


class Ideal(EoS):

    def Z(self, *args):
        return 1.

    def P(self, T, V, *args):
        return R*T/V

    def phi(self, *args):
        return 1.
