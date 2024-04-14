# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import functools
from typing import Optional

import numpy as np
from numpy import dot, exp, log
from scipy.constants import gas_constant as R

from polykin.utils.exceptions import ShapeError
from polykin.utils.math import enforce_symmetry
from polykin.utils.tools import check_bounds
from polykin.utils.types import FloatSquareMatrix, FloatVector, FloatVectorLike, FloatMatrix

from .base import PolymerActivityModel

__all__ = ['PolyNRTL', 'PolyNRTL_a']


class PolyNRTL(PolymerActivityModel):
    r"""[NRTL](https://en.wikipedia.org/wiki/Non-random_two-liquid_model)
    multicomponent activity coefficient model.

    This model is based on the following molar excess Gibbs energy
    expression:

    $$ \frac{g^{E}}{RT} = 
        \sum_i x_i \frac{\displaystyle\sum_j x_j \tau_{ji} G_{ji}}
        {\displaystyle\sum_j x_j G_{ji}} $$

    where $x_i$ are the mole fractions, $\tau_{ij}$ are the dimensionless
    interaction parameters, $\alpha_{ij}$ are the non-randomness parameters,
    and $G_{ij}=\exp(-\alpha_{ij} \tau_{ij})$. 

    In this particular implementation, the model parameters are allowed to
    depend on temperature according to the following empirical relationship
    (as done in Aspen Plus):

    \begin{aligned}
    \tau_{ij} &= a_{ij} + b_{ij}/T + e_{ij} \ln{T} + f_{ij} T \\
    \alpha_{ij} &= c_{ij} + d_{ij}(T - 273.15)
    \end{aligned}

    Moreover, $\tau_{ij}\neq\tau_{ji}$, $\tau_{ii}=0$, and
    $\alpha_{ij}=\alpha_{ji}$.

    **References**

    *   Renon, H. and Prausnitz, J.M. (1968), Local compositions in
        thermodynamic excess functions for liquid mixtures. AIChE J.,
        14: 135-144.

    Parameters
    ----------
    N : int
        Number of components.
    a : FloatSquareMatrix | None
        Matrix (NxN) of parameters, by default 0.
    b : FloatSquareMatrix | None
        Matrix (NxN) of parameters, by default 0.
    c : FloatSquareMatrix | None
        Matrix (NxN) of parameters, by default 0.3. Only the upper triangle
        must be supplied.
    d : FloatSquareMatrix | None
        Matrix (NxN) of parameters, by default 0. Only the upper triangle
        must be supplied.
    e : FloatSquareMatrix | None
        Matrix (NxN) of parameters, by default 0.
    f : FloatSquareMatrix | None
        Matrix (NxN) of parameters, by default 0.

    !!! note annotate "See also"

        * [`NRTL_gamma`](NRTL_gamma.md): related activity coefficient method.

    """

    _N: int
    _Ns: int
    _Np: int
    _Nseg: int
    _Nru: int
    _a: FloatSquareMatrix
    _b: FloatSquareMatrix
    _c: FloatSquareMatrix
    _d: FloatSquareMatrix
    _e: FloatSquareMatrix
    _f: FloatSquareMatrix
    _s: FloatVector

    def __init__(self,
                 Ns: int,
                 Np: int,
                 Nru: int,
                 a: Optional[FloatSquareMatrix] = None,
                 b: Optional[FloatSquareMatrix] = None,
                 c: Optional[FloatSquareMatrix] = None,
                 d: Optional[FloatSquareMatrix] = None,
                 e: Optional[FloatSquareMatrix] = None,
                 f: Optional[FloatSquareMatrix] = None,
                 s: Optional[FloatVector] = None
                 ) -> None:
        """Construct `NRTL` with the given parameters."""

        # Set default values
        Nseg = Ns + Nru
        if a is None:
            a = np.zeros((Nseg, Nseg))
        if b is None:
            b = np.zeros((Nseg, Nseg))
        if c is None:
            c = np.full((Nseg, Nseg), 0.3)
        if d is None:
            d = np.zeros((Nseg, Nseg))
        if e is None:
            e = np.zeros((Nseg, Nseg))
        if f is None:
            f = np.zeros((Nseg, Nseg))
        if s is None:
            s = np.ones(Nseg)

        # Check shapes
        for array in [a, b, c, d, e, f]:
            if array.shape != (Nseg, Nseg):
                raise ShapeError(
                    f"The shape of matrix {array} is invalid: {array.shape}.")

        # Check bounds (same as Aspen Plus)
        check_bounds(a, -1e2, 1e2, 'a')
        check_bounds(b, -3e4, 3e4, 'b')
        check_bounds(c, 0., 1., 'c')
        check_bounds(d, -0.02, 0.02, 'd')

        # Ensure tau_ii=0
        for array in [a, b, e, f]:
            np.fill_diagonal(array, 0.)

        # Ensure alpha_ij=alpha_ji
        for array in [c, d]:
            np.fill_diagonal(array, 0.)
            enforce_symmetry(array)

        self._N = Ns + Np
        self._Ns = Ns
        self._Np = Np
        self._Nru = Nru
        self._Nseg = Nseg
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._e = e
        self._f = f
        self._s = s

    @functools.cache
    def alpha(self,
              T: float
              ) -> FloatSquareMatrix:
        r"""Compute matrix of non-randomness parameters.

        $$ \alpha_{ij} = c_{ij} + d_{ij}(T - 273.15) $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.

        Returns
        -------
        FloatSquareMatrix (Nseg,Nseg)
            Non-randomness parameters.
        """
        return self._c + self._d*(T - 273.15)

    @functools.cache
    def tau(self,
            T: float
            ) -> FloatSquareMatrix:
        r"""Compute the matrix of dimensionless interaction parameters.

        $$ \tau_{ij} = a_{ij} + b_{ij}/T + e_{ij} \ln{T} + f_{ij} T $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.

        Returns
        -------
        FloatSquareMatrix (Nseg,Nseg)
            Dimensionless interaction parameters.
        """
        return self._a + self._b/T + self._e*log(T) + self._f*T

    def _m(self,
           DP: FloatVector,
           F: FloatMatrix,
           ) -> FloatVector:
        r"""Compute characteristic component sizes.

        Parameters
        ----------
        r : FloatMatrix (Np,Nru)
            Matrix rij with number of repeating units j in polymer i.

        Returns
        -------
        FloatVector (N)
            Characteristic component sizes.
        """
        m = np.empty(self._N)
        s = self._s
        Ns = self._Ns
        m[:Ns] = s[:Ns]
        m[Ns:] = dot(F, s[Ns:])*DP
        return m

    def gE(self,
           T: float,
           xs: FloatVector,
           DP: FloatVector,
           F: FloatMatrix
           ) -> float:
        # Residual NRTL term
        X = self._convert_xs_to_X(xs, F)
        tau = self.tau(T)
        alpha = self.alpha(T)
        G = exp(-alpha*tau)
        A = dot(X, G*tau)
        B = dot(X, G)
        gR = dot(X, A/B)  # per mole segs

        # Combinatorial FH term
        x = self._convert_xs_to_x(xs, DP)
        m = self._m(DP, F)
        gC = dot(x, log(m/dot(x, m)))
        gC /= dot(x, DP)

        return R*T*(gR + gC)

    def gamma(self,
              T: float,
              x: FloatVector
              ) -> FloatVector:
        return NRTL_gamma(x, self.tau(T), self.alpha(T))


def PolyNRTL_a(phi: FloatVector,
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
        Matrix (NxN) of interaction parameters, $\chi_{ij}$. It is expected
        (but not checked) that $\chi_{ij}=\chi_{ji}$ and $\chi_{ii}=0$.

    Returns
    -------
    FloatVector
        Activities of all components.

    !!! note annotate "See also"

        * [`FloryHuggins2_a`](FloryHuggins2_a.md): equivalent method for
          binary solvent-polymer systems.
    """
    A = dot(phi, 1/m)
    B = dot(chi, phi)
    C = 0.5*dot(phi, dot(phi, chi))
    return phi*exp(1 - m*(A - B + C))
