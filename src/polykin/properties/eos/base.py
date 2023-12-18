# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector

# import numpy as np
from numpy import log
from scipy.constants import R
from abc import ABC, abstractmethod

__all__ = ['EoS']


class EoS(ABC):

    P0 = 1e5  # Pa

    def v(self,
          T: float,
          P: float,
          y: FloatVector) -> FloatVector:
        r"""Calculate the molar volumes of the coexisting phases a fluid.

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
        FloatVector
            Molar volume of the gas and/or liquid phases. Unit = m³/mol.
        """
        return self.Z(T, P, y)*R*T/P

    @abstractmethod
    def Z(self, T: float, P: float, y: FloatVector) -> FloatVector:
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
    def P(self, T: float, v: float, y: FloatVector) -> float:
        r"""Calculate the equilibrium pressure of the fluid.

        $$ P = \frac{Z R T}{v} $$

        where $P$ is the pressure, $T$ is the temperature, $v$ is the molar
        volume, $Z$ is the compressibility factor, and $y$ is the vector of
        mole fractions.

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
    def phi(self, T: float, P: float, y: FloatVector) -> FloatVector:
        r"""Calculate the fugacity coefficients of all components in the gas
        phase.

        $$ \phi_i = \frac{f_i}{y_i P} $$

        where $\phi_i$ is the fugacity coefficient, $f_i$ is the fugacity in
        the vapor phase, $P$ is the pressure, and $y_i$ is the mole fraction in
        the gas phase.

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

    @abstractmethod
    def Ares(self, T: float, V: float, n: FloatVector) -> FloatVector:
        r"""Calculate the residual Helmholtz energy.

        $$ A-A^\circ=
        -\int_\infty^V\left(P-\frac{RT}{P}\right)dV-RT\ln{\frac{V}{V^\circ}} $$

        where the subscript $\circ$ denotes the reference state, which is the
        ideal gas state at 1 bar and $T$.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        V : float
            Volume. Unit = m³.
        n : FloatVector
            Mole amounts of all components. Unit = mol.

        Returns
        -------
        FloatVector
            Helmholtz energy departure, $A - A^\circ$. Unit = J/mol.
        """
        # TO BE updated
        pass

    def departures(self, T, P, y):
        Z = self.Z(T, P, y)
        V = Z*R*T/P
        V0 = R*T/self.P0
        Ares = self.Ares(T, V, y)
        Aig = R*T*log(V/V0)
        DA = Ares + Aig
        dT = 0.1
        Ares_plus = self.Ares(T + dT, V, y)
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
