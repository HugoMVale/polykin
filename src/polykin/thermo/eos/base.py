# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from abc import ABC, abstractmethod
from typing import Iterable

# import numpy as np
from numpy import sqrt
from scipy.constants import R

from polykin.utils.math import eps
from polykin.utils.types import FloatVector

__all__ = ['EoS']

# %%


class EoS(ABC):
    pass


# %%
class GasOrLiquidEoS(EoS):

    def v(self,
          T: float,
          P: float,
          y: FloatVector
          ) -> FloatVector:
        r"""Calculate the molar volumes of the coexisting phases a fluid.

        $$ v = \frac{Z R T}{P} $$

        where $v$ is the molar volume, $Z$ is the compressibility factor,
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
            Molar volume of the vapor and/or liquid phases. Unit = m³/mol.
        """
        return self.Z(T, P, y)*R*T/P

    def fV(self,
            T: float,
            P: float,
            y: FloatVector
           ) -> FloatVector:
        r"""Calculate the fugacity of all components in the vapor phase.

        $$ \hat{f}_i = \hat{\phi}_i y_i P $$

        $\hat{f}_i$ is the fugacity in the vapor phase, $\hat{\phi}_i(T,P,y)$
        is the fugacity coefficient, $P$ is the pressure, and $y_i$ is the mole
        fraction in the vapor phase.

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
        return self.phiV(T, P, y)*y*P

    @abstractmethod
    def P(self,
          T: float,
          v: float,
          y: FloatVector
          ) -> float:
        r"""Calculate the pressure of the fluid.

        $$ P = P(T,v,y) $$

        where $P$ is the pressure, $T$ is the temperature, $v$ is the molar
        volume, $Z$ is the compressibility factor, and $y$ is is the mole
        fraction vector.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        v : float
            Molar volume. Unit = m³/mol.
        y : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector
            Pressure. Unit = Pa.
        """
        pass

    @abstractmethod
    def Z(self,
          T: float,
          P: float,
          y: FloatVector
          ) -> FloatVector:
        r"""Calculate the compressibility factors of the coexisting phases a
        fluid.

        $$ Z = \frac{P v}{R T} $$

        where $Z$ is the compressibility factor, $P$ is the pressure, $T$ is
        the temperature, $v$ is the molar volume, and $y$ is the mole fraction
        vector.

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
            Compressibility factor of the vapor and/or liquid phases.
        """
        pass

    @abstractmethod
    def DA(self,
           T: float,
           V: float,
           n: FloatVector,
           v0: float
           ) -> float:
        r"""Calculate the departure of Helmholtz energy.

        $$ A(T,V,n) - A^{\circ}(T,V,n)$$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        V : float
            Volume. Unit = m³.
        n : FloatVector
            Mole amounts of all components. Unit = mol.
        v0 : float
            Molar volume in reference state. Unit = m³/mol.


        Returns
        -------
        FloatVector
            Helmholtz energy departure, $A - A^{\circ}$. Unit = J.
        """
        pass

    @abstractmethod
    def phiV(self,
             T: float,
             P: float,
             y: FloatVector
             ) -> FloatVector:
        r"""Calculate the fugacity coefficients of all components in the vapor
        phase.

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

    def DX(self,
           T: float,
           P: float,
           y: FloatVector,
           P0: float = 1e5
           ) -> dict[str, float]:
        v0 = R*T/P0
        nt = 1.
        n = nt*y

        Z = self.Z(T, P, y)
        if isinstance(Z, Iterable):
            Z = max(Z)  # temporary fix, get only vapor solution !!!
        V = nt*Z*R*T/P

        dT = 2*sqrt(eps)*T
        DA_minus = self.DA(T - dT, V, n, v0)
        DA_plus = self.DA(T + dT, V, n, v0)
        DA = (DA_minus + DA_plus)/2
        DS = -(DA_plus - DA_minus)/(2*dT)
        DU = DA + T*DS
        DH = DU + R*T*(Z - 1)
        DG = DA + R*T*(Z - 1)
        result = {'A': DA, 'G': DG, 'H': DH, 'S': DS, 'U': DU}
        return result


class GasEoS(GasOrLiquidEoS):

    def v(self,
          T: float,
          P: float,
          y: FloatVector
          ) -> float:
        r"""Calculate the molar volume the fluid.

        $$ v = \frac{Z R T}{P} $$

        where $v$ is the molar volue, $Z$ is the compressibility factor,
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
        float
            Molar volume of the fluid. Unit = m³/mol.
        """
        return self.Z(T, P, y)*R*T/P

    @abstractmethod
    def Z(self, T: float, P: float, y: FloatVector) -> float:
        """Calculate the compressibility factor of the fluid."""
        pass


class GasAndLiquidEoS(GasOrLiquidEoS):
    pass
