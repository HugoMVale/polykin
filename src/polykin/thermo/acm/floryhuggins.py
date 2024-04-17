# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import functools
from typing import Optional, Union

import numpy as np
from numpy import dot, exp, log
from scipy.constants import gas_constant as R

from polykin.math import derivative_complex
from polykin.utils.exceptions import ShapeError
from polykin.utils.math import enforce_symmetry
from polykin.utils.tools import check_bounds
from polykin.utils.types import (FloatArray, FloatSquareMatrix, FloatVector,
                                 Number)

# from .base import ActivityCoefficientModel

__all__ = ['FloryHuggins',
           'FloryHuggins_activity',
           'FloryHuggins2_activity']


class FloryHuggins():
    r"""[Flory-Huggings](https://en.wikipedia.org/wiki/Flory–Huggins_solution_theory)
    multicomponent activity coefficient model.

    This model is based on the following Gibbs energy of mixing per mole of
    sites:

    $$ \frac{\Delta g_{mix}}{R T}= \sum_i \frac{\phi_i}{m_i}\ln{\phi_i}
    + \sum_i \sum_{j>i} \phi_i \phi_j \chi_{ij} $$

    where $\phi_i$ are the volume, mass or segment fractions of the
    components, $\chi_{ij}$ are the interaction parameters, and $m_i$ is the
    characteristic size of the components. 

    In this particular implementation, the interaction parameters are allowed
    to depend on temperature according to the following empirical relationship
    (as used in Aspen Plus):

    $$ \chi_{ij} = a_{ij} + b_{ij}/T + c_{ij} \ln{T} + d_{ij} T + e_{ij} T^2 $$

    Moreover, $\chi_{ij}=\chi_{ji}$ and $\chi_{ii}=0$.

    **References**

    *   P.J. Flory, Principles of polymer chemistry, 1953.

    Parameters
    ----------
    N : int
        Number of components.
    a : FloatSquareMatrix | None
        Matrix (N,N) of parameters, by default 0. Only the upper triangle must
        be supplied.
    b : FloatSquareMatrix | None
        Matrix (N,N) of parameters, by default 0. Only the upper triangle must
        be supplied.
    c : FloatSquareMatrix | None
        Matrix (N,N) of parameters, by default 0. Only the upper triangle must
        be supplied.
    d : FloatSquareMatrix | None
        Matrix (N,N) of parameters, by default 0. Only the upper triangle must
        be supplied.
    e : FloatSquareMatrix | None
        Matrix (N,N) of parameters, by default 0. Only the upper triangle must
        be supplied.

    """

    _N: int
    _a: FloatSquareMatrix
    _b: FloatSquareMatrix
    _c: FloatSquareMatrix
    _d: FloatSquareMatrix
    _e: FloatSquareMatrix

    def __init__(self,
                 N: int,
                 a: Optional[FloatSquareMatrix] = None,
                 b: Optional[FloatSquareMatrix] = None,
                 c: Optional[FloatSquareMatrix] = None,
                 d: Optional[FloatSquareMatrix] = None,
                 e: Optional[FloatSquareMatrix] = None
                 ) -> None:
        """Construct `FloryHuggins` with the given parameters."""

        # Set default values
        if a is None:
            a = np.zeros((N, N))
        if b is None:
            b = np.zeros((N, N))
        if c is None:
            c = np.zeros((N, N))
        if d is None:
            d = np.zeros((N, N))
        if e is None:
            e = np.zeros((N, N))

        # Check shapes
        for array in [a, b, c, d, e]:
            if array.shape != (N, N):
                raise ShapeError(
                    f"The shape of matrix {array} is invalid: {array.shape}.")

        # Check bounds (same as Aspen Plus)
        check_bounds(a, -1e2, 1e2, 'a')
        check_bounds(b, -1e6, 1e6, 'b')
        check_bounds(c, -1e6, 1e6, 'c')
        check_bounds(d, -1e6, 1e6, 'd')
        check_bounds(e, -1e6, 1e6, 'e')

        # Ensure chi_ii=0 and chi_ij=chi_ji
        for array in [a, b, c, d, e]:
            np.fill_diagonal(array, 0.)
            enforce_symmetry(array)

        self._N = N
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._e = e

    @functools.cache
    def chi(self, T: float) -> FloatSquareMatrix:
        r"""Compute the matrix of interaction parameters.

        $$
        \chi_{ij} = a_{ij} + b_{ij}/T + c_{ij} \ln{T} + d_{ij} T + e_{ij} T^2
        $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.

        Returns
        -------
        FloatSquareMatrix
            Matrix of interaction parameters.
        """
        return self._a + self._b/T + self._c*log(T) + self._d*T + self._e*T**2

    def Dgmix(self,
              T: Number,
              phi: FloatVector,
              m: FloatVector) -> Number:
        r"""Gibbs energy of mixing per mole of sites, $\Delta g_{mix}$.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        phi : FloatVector
            Volume, mass or segment fractions of all components.
        m : FloatVector
            Characteristic size of all components, typically equal to 1 for
            small molecules and equal to the average degree of polymerization
            for polymers.

        Returns
        -------
        float
            Gibbs energy of mixing per mole of sites. Unit = J/mol.
        """
        p = phi > 0.
        gC = dot(phi[p]/m[p], log(phi[p]))
        gR = 0.5*dot(phi, dot(phi, self.chi(T)))
        return R*T*(gC + gR)

    def Dhmix(self,
              T: float,
              phi: FloatVector,
              m: FloatVector) -> float:
        r"""Enthalpy of mixing per mole of sites, $\Delta h_{mix}$.

        $$ \Delta h_{mix} = \Delta g_{mix} + T \Delta s_{mix} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        phi : FloatVector
            Volume, mass or segment fractions of all components.
        m : FloatVector
            Characteristic size of all components, typically equal to 1 for
            small molecules and equal to the average degree of polymerization
            for polymers.

        Returns
        -------
        float
            Enthalpy of mixing per mole of sites. Unit = J/mol.
        """
        return self.Dgmix(T, phi, m) + T*self.Dsmix(T, phi, m)

    def Dsmix(self,
              T: float,
              phi: FloatVector,
              m: FloatVector) -> float:
        r"""Entropy of mixing per mole of sites, $\Delta s_{mix}$.

        $$ \Delta s_{mix} = -\left(\frac{\partial \Delta g_{mix}}
           {\partial T}\right)_{P,\phi_i,m_i} $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        phi : FloatVector
            Volume, mass or segment fractions of all components.
        m : FloatVector
            Characteristic size of all components, typically equal to 1 for
            small molecules and equal to the average degree of polymerization
            for polymers.

        Returns
        -------
        float
            Entropy of mixing per mole of sites. Unit = J/(mol·K).
        """
        return -derivative_complex(lambda t: self.Dgmix(t, phi, m), T)[0]

    def a(self,
          T: float,
          phi: FloatVector,
          m: FloatVector
          ) -> FloatVector:
        r"""Activities, $a_i$.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        phi : FloatVector
            Volume, mass or segment fractions of all components.
        m : FloatVector
            Characteristic size of all components, typically equal to 1 for
            small molecules and equal to the average degree of polymerization
            for polymers.

        Returns
        -------
        FloatVector
            Activities of all components.
        """
        return FloryHuggins_activity(phi, m, self.chi(T))


def FloryHuggins_activity(phi: FloatVector,
                          m: FloatVector,
                          chi: FloatSquareMatrix
                          ) -> FloatVector:
    r"""Calculate the activities of a multicomponent mixture according to the
    Flory-Huggins model.

    $$ 
    \ln{a_i} = \ln{\phi_i} + 1 - m_i \left(\sum_j \frac{\phi_j}{m_j} - 
    \sum_j \phi_j \chi_{ij} + \sum_j \sum_{k>j} \phi_j \phi_k \chi_{jk} \right)
    $$

    where $\phi_i$ are the volume, mass or segment fractions of the
    components, $\chi_{ij}$ are the interaction parameters, and $m_i$ is the
    characteristic size of the components. 

    !!! note

        The Flory-Huggins model was originally formulated using the volume
        fraction as primary composition variable, but many authors prefer mass
        fractions in order to skip issues with densities, etc. This equation
        can be used in all cases, as long as $\phi$ is taken to be the same
        variable used to estimate $\chi$.
        In this particular implementation, the matrix of interaction parameters
        is independent of molecular size and, thus, symmetric.

    **References**

    *   P.J. Flory, Principles of polymer chemistry, 1953.
    *   Lindvig, Thomas, et al. "Modeling of multicomponent vapor-liquid
        equilibria for polymer-solvent systems." Fluid phase equilibria 220.1
        (2004): 11-20.

    Parameters
    ----------
    phi : FloatVector
        Volume, mass or segment fractions of all components, $\phi_i$. The
        composition variable must match the one used in the determination of
        $\chi_{ij}$.
    m : FloatVector
        Characteristic size of all components, typically equal to 1 for small
        molecules and equal to the average degree of polymerization for
        polymers.
    chi : FloatSquareMatrix
        Matrix (N,N) of interaction parameters, $\chi_{ij}$. It is expected
        (but not checked) that $\chi_{ij}=\chi_{ji}$ and $\chi_{ii}=0$.

    Returns
    -------
    FloatVector
        Activities of all components.

    See also
    --------
    * [`FloryHuggins2_activity`](FloryHuggins2_activity.md): equivalent
      method for binary solvent-polymer systems.

    """
    A = dot(phi, 1/m)
    B = dot(chi, phi)
    C = 0.5*dot(phi, dot(phi, chi))
    return phi*exp(1 - m*(A - B + C))


def FloryHuggins2_activity(phi1: Union[float, FloatArray],
                           m: Union[float, FloatArray],
                           chi: Union[float, FloatArray]
                           ) -> Union[float, FloatArray]:
    r"""Calculate the solvent activity of a binary polymer solution according
    to the Flory-Huggins model.

    $$ \ln{a_1} = \ln \phi_1 + \left(\ 1 - \frac{1}{m} \right)(1-\phi_1)
                  + \chi_{12}(1-\phi_1)^2 $$

    where $\phi_1$ is the volume, mass or segment fraction of the solvent in
    the solution, $\chi$ is the solvent-polymer interaction parameter, and $m$
    is the ratio of the molar volume of the polymer to the solvent, often
    approximated as the degree of polymerization. 

    !!! note

        The Flory-Huggins model was originally formulated using the volume
        fraction as primary composition variable, but many authors prefer mass
        fractions in order to skip issues with densities, etc. This equation
        can be used in all cases, as long as $\phi_1$ is taken to be the same
        variable used to estimate $\chi$. For example, if a given publication
        reports values of $\chi$ estimated with the mass fraction version of
        the Flory-Huggins model, then $\phi_1$ should be the mass fraction of
        solvent.

    **References**

    *   P.J. Flory, Principles of polymer chemistry, 1953.

    Parameters
    ----------
    phi1 : float | FloatArray
        Volume, mass, or segment fraction of solvent in the solution, $\phi_1$.
        The composition variable must match the one used in the determination
        of $\chi$.
    m : float | FloatArray
        Ratio of the molar volume of the polymer chain to the solvent, often
        approximated as the degree of polymerization.
    chi : float | FloatArray
        Solvent-polymer interaction parameter, $\chi$.

    Returns
    -------
    FloatVector
        Activity of the solvent.

    See also
    --------
    * [`FloryHuggins_activity`](FloryHuggins_activity.md): equivalent method
      for multicomponent systems.

    Examples
    --------
    Calculate the activity of ethylbenzene in a ethylbenzene-polybutadiene
    solution with 25 wt% solvent content. Assume $\chi=0.29$.
    >>> from polykin.thermo.acm import FloryHuggins2_gamma
    >>> gamma = FloryHuggins2_gamma(phi1=0.25, m=1e6, chi=0.29)
    >>> print(f"{gamma:.2f}")
    0.62

    """
    phi2 = 1 - phi1
    return phi1*exp((1 - 1/m)*phi2 + chi*phi2**2)
