# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from abc import ABC, abstractmethod

import numpy as np
from numpy import dot, log
from scipy.constants import R

from polykin.math import derivative_complex
from polykin.utils.types import FloatMatrix, FloatVector, Number


class ActivityModel(ABC):

    _N: int
    name: str

    def __init__(self,
                 N: int,
                 name: str
                 ) -> None:
        self._N = N
        self.name = name

    @property
    def N(self) -> int:
        "Number of components."
        return self._N

    def _Dsmix_ideal(self,
                     T: float,
                     x: FloatVector
                     ) -> float:
        r"""Molar entropy of mixing of ideal solution, $\Delta s_{mix}^{ideal}$.

        $$ \Delta s_{mix}^{ideal} = - R \sum_i {x_i \ln{x_i}} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector (N)
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar entropy of mixing. Unit = J/(mol·K).
        """
        p = x > 0
        return -R*dot(x[p], log(x[p]))


class SmallSpeciesActivityModel(ActivityModel):

    def Dgmix(self,
              T: float,
              x: FloatVector
              ) -> float:
        r"""Molar Gibbs energy of mixing, $\Delta g_{mix}$.

        $$ \Delta g_{mix} = \Delta h_{mix} -T \Delta s_{mix} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector (N)
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar Gibbs energy of mixing. Unit = J/mol.
        """
        return self.gE(T, x) - T*self._Dsmix_ideal(T, x)

    def Dhmix(self,
              T: float,
              x: FloatVector
              ) -> float:
        r"""Molar enthalpy of mixing, $\Delta h_{mix}$.

        $$ \Delta h_{mix} = h^{E} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector (N)
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar enthalpy of mixing. Unit = J/mol.
        """
        return self.hE(T, x)

    def Dsmix(self,
              T: float,
              x: FloatVector
              ) -> float:
        r"""Molar entropy of mixing, $\Delta s_{mix}$.

        $$ \Delta s_{mix} = s^{E} - R \sum_i {x_i \ln{x_i}} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector (N)
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar entropy of mixing. Unit = J/(mol·K).
        """
        return self.sE(T, x) + self._Dsmix_ideal(T, x)

    def hE(self,
           T: float,
           x: FloatVector
           ) -> float:
        r"""Molar excess enthalpy, $h^{E}$.

        $$ h^{E} = g^{E} + T s^{E} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector (N)
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar excess enthalpy. Unit = J/mol.
        """
        return self.gE(T, x) + T*self.sE(T, x)

    def sE(self,
           T: float,
           x: FloatVector
           ) -> float:
        r"""Molar excess entropy, $s^{E}$.

        $$ s^{E} = -\left(\frac{\partial g^{E}}{\partial T}\right)_{P,x_i} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector (N)
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar excess entropy. Unit = J/(mol·K).
        """
        return -1*derivative_complex(lambda t: self.gE(t, x), T)[0]

    def activity(self,
                 T: float,
                 x: FloatVector
                 ) -> FloatVector:
        r"""Activities, $a_i$.

        $$ a_i = x_i \gamma_i $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector (N)
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector (N)
            Activities of all components.
        """
        return x*self.gamma(T, x)

    @abstractmethod
    def gE(self,
           T: Number,
           x: FloatVector
           ) -> Number:
        r"""Molar excess Gibbs energy, $g^{E}$.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector (N)
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar excess Gibbs energy. Unit = J/mol.
        """
        pass

    @abstractmethod
    def gamma(self,
              T: float,
              x: FloatVector
              ) -> FloatVector:
        r"""Activity coefficients based on mole fraction, $\gamma_i$.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector (N)
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector (N)
            Activity coefficients of all components.
        """
        pass


class PolymerActivityModel(ActivityModel):

    def Dgmix(self,
              T: float,
              xs: FloatVector,
              DP: FloatVector,
              F: FloatMatrix
              ) -> float:
        r"""Molar Gibbs energy of mixing, $\Delta g_{mix}$.

        $$ \Delta g_{mix} = \Delta h_{mix} -T \Delta s_{mix} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        xs : FloatVector (N)
            Segment-based mole fractions of all components. Unit = mol/mol.
        DP : FloatVector (Np)
            Degree of polymerization of all polymer components. 
        F : FloatMatrix (Np, Nru)
            Composition all polymer components.

        Returns
        -------
        float
            Gibbs energy of mixing per mole of segments. Unit = J/mol.
        """
        x = 0  # to be done
        return self.gE(T, xs, DP, F) - T*self._Dsmix_ideal(T, x)

    def Dhmix(self,
              T: float,
              xs: FloatVector,
              DP: FloatVector,
              F: FloatMatrix
              ) -> float:
        r"""Molar enthalpy of mixing, $\Delta h_{mix}$.

        $$ \Delta h_{mix} = h^{E} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        xs : FloatVector (N)
            Segment-based mole fractions of all components. Unit = mol/mol.
        DP : FloatVector (Np)
            Degree of polymerization of all polymer components. 
        F : FloatMatrix (Np, Nru)
            Composition all polymer components.

        Returns
        -------
        float
            Enthalpy of mixing per mole of segments. Unit = J/mol.
        """
        return self.hE(T, xs, DP, F)

    def Dsmix(self,
              T: float,
              xs: FloatVector,
              DP: FloatVector,
              F: FloatMatrix
              ) -> float:
        r"""Molar entropy of mixing, $\Delta s_{mix}$.

        $$ \Delta s_{mix} = s^{E} - R \sum_i {x_i \ln{x_i}} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        xs : FloatVector (N)
            Segment-based mole fractions of all components. Unit = mol/mol.
        DP : FloatVector (Np)
            Degree of polymerization of all polymer components. 
        F : FloatMatrix (Np, Nru)
            Composition all polymer components.

        Returns
        -------
        float
            Entropy of mixing per mole of segments. Unit = J/(mol·K).
        """
        x = 0  # fix!!
        return self.sE(T, xs, DP, F) + self._Dsmix_ideal(T, x)

    def hE(self,
           T: float,
           xs: FloatVector,
           DP: FloatVector,
           F: FloatMatrix
           ) -> float:
        r"""Molar excess enthalpy, $h^{E}$.

        $$ h^{E} = g^{E} + T s^{E} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        xs : FloatVector (N)
            Segment-based mole fractions of all components. Unit = mol/mol.
        DP : FloatVector (Np)
            Degree of polymerization of all polymer components. 
        F : FloatMatrix (Np, Nru)
            Composition all polymer components.

        Returns
        -------
        float
            Excess enthalpy per mole of segments. Unit = J/mol.
        """
        return self.gE(T, xs, DP, F) + T*self.sE(T, xs, DP, F)

    def sE(self,
           T: float,
           xs: FloatVector,
           DP: FloatVector,
           F: FloatMatrix
           ) -> float:
        r"""Molar excess entropy, $s^{E}$.

        $$ s^{E} = -\left(\frac{\partial g^{E}}{\partial T}\right)_{P,x_i} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        xs : FloatVector (N)
            Segment-based mole fractions of all components. Unit = mol/mol.
        DP : FloatVector (Np)
            Degree of polymerization of all polymer components. 
        F : FloatMatrix (Np, Nru)
            Composition all polymer components.

        Returns
        -------
        float
            Excess entropy per mole of segments. Unit = J/(mol·K).
        """
        return -1*derivative_complex(lambda t: self.gE(t, xs, DP, F), T)[0]

    @staticmethod
    def _convert_xs_to_x(xs: FloatVector,
                         DP: FloatVector
                         ) -> FloatVector:
        """Convert segment-based mole fractions to mole fractions.

        Parameters
        ----------
        xs : FloatVector (N)
            Segment-based mole fractions of all components. Unit = mol/mol.
        DP : FloatVector (Np)
            Degree of polymerization of all polymer components. 

        Returns
        -------
        FloatVector (N)
            Mole fractions of all components. Unit = mol/mol.
        """
        x = xs/DP
        x /= x.sum()
        return x

    @staticmethod
    def _convert_xs_to_X(xs: FloatVector,
                         F: FloatMatrix,
                         ) -> FloatVector:
        """Convert segment-based component mole fractions to segment mole
        fractions.

        Parameters
        ----------
        xs : FloatVector (N)
            Segment-based mole fractions of all components. Unit = mol/mol.
        F : FloatMatrix (Np, Nru)
            Composition all polymer components.

        Returns
        -------
        FloatVector (Nseg)
            Mole fractions of all segments. Unit = mol/mol.
        """
        N = xs.shape[0]
        Np, Nru = F.shape
        Ns = N - Np
        Nseg = Ns + Nru
        X = np.empty(Nseg)
        X[:Ns] = xs[:Ns]
        X[Ns:] = dot(xs[Ns:], F)
        X /= X.sum()
        return X

    @abstractmethod
    def gE(self,
           T: Number,
           xs: FloatVector,
           DP: FloatVector,
           F: FloatMatrix
           ) -> Number:
        r"""Molar excess Gibbs energy, $g^{E}$.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        xs : FloatVector (N)
            Segment-based mole fractions of all components. Unit = mol/mol.
        DP : FloatVector (Np)
            Degree of polymerization of all polymer components. 
        F : FloatMatrix (Np, Nru)
            Composition all polymer components.

        Returns
        -------
        float
            Excess Gibbs energy per mole of segments. Unit = J/mol.
        """
        pass

    @abstractmethod
    def activity(self,
                 T: float,
                 xs: FloatVector,
                 DP: FloatVector,
                 F: FloatMatrix
                 ) -> FloatVector:
        r"""Activities, $\a_i$.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        xs : FloatVector (N)
            Segment-based mole fractions of all components. Unit = mol/mol.
        DP : FloatVector (Np)
            Degree of polymerization of all polymer components. 
        F : FloatMatrix (Np, Nru)
            Composition all polymer components.

        Returns
        -------
        FloatVector (N)
            Activities of all components.
        """
        pass
