# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from abc import abstractmethod
from typing import Union

import numpy as np
from numpy import log
from scipy.optimize import root_scalar

from polykin.utils.math import vectorize
from polykin.utils.tools import check_bounds
from polykin.utils.types import FloatArray

from .base import PolymerPVTEquation

__all__ = ['Flory',
           'HartmannHaque',
           'SanchezLacombe'
           ]

# %% PolymerPVTEoS


class PolymerPVTEoS(PolymerPVTEquation):
    r"""_Abstract_ polymer equation of state in reduced form,
    $V(T, P, V0, T0, P0)$."""

    def __init__(self,
                 V0: float,
                 T0: float,
                 P0: float,
                 Tmin: float = 0.0,
                 Tmax: float = np.inf,
                 Pmin: float = 0.0,
                 Pmax: float = np.inf,
                 name: str = ''
                 ) -> None:

        # Check bounds
        check_bounds(V0, 0, np.inf, 'V0')
        check_bounds(T0, 0, np.inf, 'T0')
        check_bounds(P0, 0, np.inf, 'P0')

        self.V0 = V0
        self.T0 = T0
        self.P0 = P0
        super().__init__(Tmin, Tmax, Pmin, Pmax, name)

    @vectorize
    def eval(self,
             T: Union[float, FloatArray],
             P: Union[float, FloatArray]
             ) -> Union[float, FloatArray]:
        r"""Evaluate specific volume, $\hat{V}$, at given SI conditions without
        unit conversions or checks.

        Parameters
        ----------
        T : float | FloatArray
            Temperature.
            Unit = K.
        P : float | FloatArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        float | FloatArray
            Specific volume.
            Unit = m³/kg.
        """
        t = T/self.T0
        p = P/self.P0
        solution = root_scalar(f=self.equation,
                               args=(t, p),
                               # bracket=[1.1, 1.5],
                               x0=1.05,
                               method='halley',
                               fprime=True,
                               fprime2=True)

        if solution.converged:
            v = solution.root
            V = v*self.V0
        else:
            print(solution.flag)
            V = -1.
        return V

    def alpha(self,
              T: Union[float, FloatArray],
              P: Union[float, FloatArray]
              ) -> Union[float, FloatArray]:
        r"""Calculate thermal expansion coefficient, $\alpha$.

        $$\alpha=\frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_{P}$$

        Parameters
        ----------
        T : float | FloatArray
            Temperature.
            Unit = K.
        P : float | FloatArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        float | FloatArray
            Thermal expansion coefficient, $\alpha$.
            Unit = 1/K.
        """
        dT = 0.5
        V2 = self.eval(T + dT, P)
        V1 = self.eval(T - dT, P)
        return (V2 - V1)/dT/(V1 + V2)

    def beta(self,
             T: Union[float, FloatArray],
             P: Union[float, FloatArray]
             ) -> Union[float, FloatArray]:
        r"""Calculate isothermal compressibility coefficient, $\beta$.

        $$\beta=-\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_{T}$$

        Parameters
        ----------
        T : float | FloatArray
            Temperature.
            Unit = K.
        P : float | FloatArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        float | FloatArray
            Isothermal compressibility coefficient, $\beta$.
            Unit = 1/Pa.
        """
        dP = 1e5
        P2 = P + dP
        P1 = np.max(P - dP, 0)
        V2 = self.eval(T, P2)
        V1 = self.eval(T, P1)
        return -(V2 - V1)/(P2 - P1)/(V1 + V2)*2

    @staticmethod
    @abstractmethod
    def equation(v: float, t: float, p: float) -> tuple[float, ...]:
        """Equation of state and its volume derivatives.

        Parameters
        ----------
        v : float
            Reduced volume.
        t : float
            Reduced temperature.
        p : float
            Reduced pressure.

        Returns
        -------
        tuple[float,...]
            Equation of state, first derivative, second derivative.
        """
        pass

# %% Flory


class Flory(PolymerPVTEoS):
    r"""Flory equation of state for the specific volume of a polymer.

    This EoS implements the following implicit PVT dependence:

    $$ \frac{\tilde{P}\tilde{V}}{\tilde{T}} =
      \frac{\tilde{V}^{1/3}}{\tilde{V}^{1/3}-1}-\frac{1}{\tilde{V}\tilde{T}}$$

    where $\tilde{V}=V/V^*$, $\tilde{P}=P/P^*$ and $\tilde{T}=T/T^*$ are,
    respectively, the reduced volume, reduced pressure and reduced temperature.
    $V^*$, $P^*$ and $T^*$ are reference quantities that are polymer dependent.

    **References**

    *   Caruthers et al. Handbook of Diffusion and Thermal Properties of
        Polymers and Polymer Solutions. AIChE, 1998.

    Parameters
    ----------
    V0 : float
        Reference volume, $V^*$.
    T0 : float
        Reference temperature, $T^*$.
    P0 : float
        Reference pressure, $P^*$.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    Pmin : float
        Lower pressure bound.
        Unit = Pa.
    Pmax : float
        Upper pressure bound.
        Unit = Pa.
    name : str
        Name.
    """

    @staticmethod
    def equation(v: float,
                 t: float,
                 p: float
                 ) -> tuple[float, float, float]:
        r"""Flory equation of state and its volume derivatives.

        Parameters
        ----------
        v : float
            Reduced volume, $\tilde{V}$.
        t : float
            Reduced temperature, $\tilde{T}$.
        p : float
            Reduced pressure, $\tilde{P}$.

        Returns
        -------
        tuple[float, float, float]
            Equation of state, first derivative, second derivative.
        """
        f = p*v/t - (v**(1/3)/(v**(1/3) - 1) - 1/(v*t))  # =0
        d1f = p/t - 1/(t*v**2) - 1/(3*(v**(1/3) - 1)*v**(2/3)) + \
            1/(3*(v**(1/3) - 1)**2*v**(1/3))
        d2f = (2*(9/t + (v**(4/3) - 2*v**(5/3))/(-1 + v**(1/3))**3))/(9*v**3)
        return (f, d1f, d2f)

# %% HartmannHaque


class HartmannHaque(PolymerPVTEoS):
    r"""Hartmann-Haque equation of state for the specific volume of a polymer.

    This EoS implements the following implicit PVT dependence:

    $$ \tilde{P}\tilde{V}^5=\tilde{T}^{3/2}-\ln{\tilde{V}} $$

    where $\tilde{V}=V/V^*$, $\tilde{P}=P/P^*$ and $\tilde{T}=T/T^*$ are,
    respectively, the reduced volume, reduced pressure and reduced temperature.
    $V^*$, $P^*$ and $T^*$ are reference quantities that are polymer dependent.

    **References**

    *   Caruthers et al. Handbook of Diffusion and Thermal Properties of
        Polymers and Polymer Solutions. AIChE, 1998.

    Parameters
    ----------
    V0 : float
        Reference volume, $V^*$.
    T0 : float
        Reference temperature, $T^*$.
    P0 : float
        Reference pressure, $P^*$.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    Pmin : float
        Lower pressure bound.
        Unit = Pa.
    Pmax : float
        Upper pressure bound.
        Unit = Pa.
    name : str
        Name.
    """

    @staticmethod
    def equation(v: float,
                 t: float,
                 p: float
                 ) -> tuple[float, float, float]:
        """Hartmann-Haque equation of state and its volume derivatives.

        Parameters
        ----------
        v : float
            Reduced volume.
        t : float
            Reduced temperature.
        p : float
            Reduced pressure.

        Returns
        -------
        tuple[float, float, float]
            Equation of state, first derivative, second derivative.
        """
        f = p*v**5 - t**(3/2) + log(v)  # =0
        d1f = 5*p*v**4 + 1/v
        d2f = 20*p*v**3 - 1/v**2
        return (f, d1f, d2f)

# %% SanchezLacombe


class SanchezLacombe(PolymerPVTEoS):
    r"""Sanchez-Lacombe equation of state for the specific volume of a polymer.

    This EoS implements the following implicit PVT dependence:

    $$ \frac{1}{\tilde{V}^2} + \tilde{P} +
        \tilde{T}\left [ \ln\left ( 1-\frac{1}{\tilde{V}} \right ) +
        \frac{1}{\tilde{V}} \right ]=0 $$

    where $\tilde{V}=V/V^*$, $\tilde{P}=P/P^*$ and $\tilde{T}=T/T^*$ are,
    respectively, the reduced volume, reduced pressure and reduced temperature.
    $V^*$, $P^*$ and $T^*$ are reference quantities that are polymer dependent.

    **References**

    *   Caruthers et al. Handbook of Diffusion and Thermal Properties of
        Polymers and Polymer Solutions. AIChE, 1998.

    Parameters
    ----------
    V0 : float
        Reference volume, $V^*$.
    T0 : float
        Reference temperature, $T^*$.
    P0 : float
        Reference pressure, $P^*$.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    Pmin : float
        Lower pressure bound.
        Unit = Pa.
    Pmax : float
        Upper pressure bound.
        Unit = Pa.
    name : str
        Name.
    """

    @staticmethod
    def equation(v: float,
                 t: float,
                 p: float
                 ) -> tuple[float, float, float]:
        """Sanchez-Lacombe equation of state and its volume derivatives.

        Parameters
        ----------
        v : float
            Reduced volume.
        t : float
            Reduced temperature.
        p : float
            Reduced pressure.

        Returns
        -------
        tuple[float, float, float]
            Equation of state, first derivative, second derivative.
        """
        f = 1/v**2 + p + t*(log(1 - 1/v) + 1/v)  # =0
        d1f = ((t - 2)*v + 2)/((v - 1)*v**3)
        d2f = (-3*(t - 2)*v**2 + 2*(t - 6)*v + 6)/((v - 1)**2*v**4)
        return (f, d1f, d2f)
