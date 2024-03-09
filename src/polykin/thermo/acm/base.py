# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from abc import ABC, abstractmethod

import numpy as np
from numpy import log, sqrt
from scipy.constants import R

from polykin.utils.math import eps
from polykin.utils.types import FloatVector


class ActivityCoefficientModel(ABC):

    def Dgmix(self, T: float, x: FloatVector) -> float:
        r"""Molar Gibbs energy of mixing, $\Delta g_{mix}$.

        $$ \Delta g_{mix} = \Delta h_{mix} -T \Delta s_{mix} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar Gibbs energy of mixing. Unit = J/mol.
        """
        return self.Dhmix(T, x) - T*self.Dsmix(T, x)

    def Dsmix(self, T: float, x: FloatVector) -> float:
        r"""Molar entropy of mixing, $\Delta s_{mix}$.

        $$ \Delta s_{mix} = s^{E} + R \sum_i {x_i \ln{x_i}} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar entropy of mixing. Unit = J/(mol·K).
        """
        return self.sE(T, x) + R*np.dot(x, log(x))

    def Dhmix(self, T: float, x: FloatVector) -> float:
        r"""Molar enthalpy of mixing, $\Delta h_{mix}$.

        $$ \Delta h_{mix} = h^{E} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar enthalpy of mixing. Unit = J/mol.
        """
        return self.hE(T, x)

    def hE(self, T: float, x: FloatVector) -> float:
        r"""Molar excess enthalpy, $h^{E}$.

        $$ h^{E} = g^{E} + T s^{E} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar excess enthalpy. Unit = J/mol.
        """
        return self.gE(T, x) + T*self.sE(T, x)

    def sE(self, T: float, x: FloatVector) -> float:
        r"""Molar excess entropy, $s^{E}$.

        $$ s^{E} = -\left(\frac{\partial g^{E}}{\partial T}\right)_{P,x_i} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar excess entropy. Unit = J/(mol·K).
        """
        dT = 2*sqrt(eps)*T
        gE_plus = self.gE(T + dT, x)
        gE_minus = self.gE(T - dT, x)
        return (gE_minus - gE_plus)/(2*dT)

    def a(self, T: float, x: FloatVector) -> FloatVector:
        r"""Activities, $a_i$.

        $$ a_i = x_i \gamma_i $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector
            Activities.
        """
        return x*self.gamma(T, x)

    @abstractmethod
    def gE(self, T: float, x: FloatVector) -> float:
        r"""Molar excess Gibbs energy, $g^{E}$.

        Parameters
        ----------
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.
        T : float
            Temperature. Unit = K.

        Returns
        -------
        float
            Molar excess Gibbs energy. Unit = J/mol.
        """
        pass

    @abstractmethod
    def gamma(self, T: float, x: FloatVector) -> FloatVector:
        r"""Activity coefficients, $\gamma_i$.

        Parameters
        ----------
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.
        T : float
            Temperature. Unit = K.

        Returns
        -------
        FloatVector
            Activity coefficients.
        """
        pass
