# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector

# import numpy as np
from numpy import log
from scipy.constants import R
from abc import ABC, abstractmethod

__all__ = ['EoS']

# %%


class EoS(ABC):
    pass


# %%
class GasOrLiquidEoS(EoS):

    P0 = 1e5  # Pa

    def v(self,
          T: float,
          P: float,
          y: FloatVector) -> FloatVector:
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
            Molar volume of the gas and/or liquid phases. Unit = m³/mol.
        """
        return self.Z(T, P, y)*R*T/P

    def fv(self,
            T: float,
            P: float,
            y: FloatVector
           ) -> FloatVector:
        r"""Calculate the fugacity of all components in the gas phase.

        $$ \hat{f}_i = \hat{\phi}_i y_i P $$

        $\hat{f}_i$ is the fugacity in the gas phase, $\hat{\phi}_i(T,P,y)$ is
        the fugacity coefficient, $P$ is the pressure, and $y_i$ is the mole
        fraction in the gas phase.

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
        return self.phi(T, P, y)*y*P

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
            Compressibility factor of the gas and/or liquid phases.
        """
        pass

    @abstractmethod
    def DA(self,
           n: FloatVector,
           T: float,
           V: float,
           V0: float
           ) -> float:
        r"""Calculate the residual Helmholtz energy.

        $$ A-A^\circ=
        -\int_\infty^V\left(P-\frac{RT}{P}\right)dV-RT\ln{\frac{V}{V^\circ}} $$

        where the subscript $\circ$ denotes the reference state, which is the
        ideal gas state at 1 bar and $T$.

        Parameters
        ----------
        n : FloatVector
            Mole amounts of all components. Unit = mol.
        T : float
            Temperature. Unit = K.
        V : float
            Volume. Unit = m³.
        V0 : float
            Reference volume. Unit = m³.

        Returns
        -------
        FloatVector
            Helmholtz energy departure, $A - A^\circ$. Unit = J/mol.
        """
        pass

    @abstractmethod
    def phi(self,
            T: float,
            P: float,
            y: FloatVector
            ) -> FloatVector:
        r"""Calculate the fugacity coefficients of all components in the gas
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

    def departures(self, T, P, y):
        Z = self.Z(T, P, y)
        V = Z*R*T/P
        V0 = R*T/self.P0
        Ares = self.DA(T, V, y)
        Aig = R*T*log(V/V0)
        DA = Ares + Aig
        dT = 0.1
        Ares_plus = self.DA(T + dT, V, y)
        DS = (Ares_plus - Ares)/dT + Aig/T
        DU = DA + T*DS
        DH = DA + T*DS + R*T*(Z - 1.)
        DG = DA + R*T*(Z - 1.)
        result = {
            'A': DA,
            'G': DG,
            'H': DH,
            'S': DS,
            'U': DU}
        return result


class GasEoS(GasOrLiquidEoS):

    def v(self,
          T: float,
          P: float,
          y: FloatVector) -> float:
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
