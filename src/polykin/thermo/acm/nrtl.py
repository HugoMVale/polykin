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
from polykin.utils.types import FloatSquareMatrix, FloatVector

from .base import ACM

__all__ = ['NRTL', 'NRTL_gamma']


class NRTL(ACM):
    r"""[NRTL](https://en.wikipedia.org/wiki/Non-random_two-liquid_model)
    activity coefficient model.

    This model is based on the following molar excess Gibbs energy
    expression:

    $$ \frac{g^{E}}{RT} = 
        \sum_i x_i \frac{\displaystyle\sum_j x_j \tau_{ji} G_{ji}}
        {\displaystyle\sum_j x_j G_{ji}} $$

    where $x_i$ are the mole fractions, $\tau_{ij}$ are the dimensionless
    interaction parameters, $\alpha_{ij}$ are the non-randomness parameters,
    and $G_{ij}=\exp(-\alpha_{ij} \tau_{ij})$. Moreover,
    $\tau_{ij}\neq\tau_{ji}$, $\tau_{ii}=0$, and $\alpha_{ij}=\alpha_{ji}$.

    In this particular implementation, the model parameters are allowed to
    depend on temperature according to the following empirical relationship
    (as done in Aspen Plus):

    \begin{aligned}
    \tau_{ij} &= a_{ij} + b_{ij}/T + e_{ij} \ln{T} + f_{ij} T \\
    \alpha_{ij} &= c_{ij} + d_{ij}(T - 273.15)
    \end{aligned}

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
    _a: FloatSquareMatrix
    _b: FloatSquareMatrix
    _c: FloatSquareMatrix
    _d: FloatSquareMatrix
    _e: FloatSquareMatrix
    _f: FloatSquareMatrix

    def __init__(self,
                 N: int,
                 a: Optional[FloatSquareMatrix] = None,
                 b: Optional[FloatSquareMatrix] = None,
                 c: Optional[FloatSquareMatrix] = None,
                 d: Optional[FloatSquareMatrix] = None,
                 e: Optional[FloatSquareMatrix] = None,
                 f: Optional[FloatSquareMatrix] = None
                 ) -> None:
        """Construct `NRTL` with the given parameters."""

        # Set default values
        if a is None:
            a = np.zeros((N, N))
        if b is None:
            b = np.zeros((N, N))
        if c is None:
            c = np.full((N, N), 0.3)
        if d is None:
            d = np.zeros((N, N))
        if e is None:
            e = np.zeros((N, N))
        if f is None:
            f = np.zeros((N, N))

        # Check shapes
        for array in [a, b, c, d, e, f]:
            if array.shape != (N, N):
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

        self._N = N
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._e = e
        self._f = f

    @functools.cache
    def alpha(self, T: float) -> FloatSquareMatrix:
        r"""Compute matrix of non-randomness parameters.

        $$ \alpha_{ij} = c_{ij} + d_{ij}(T - 273.15) $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.

        Returns
        -------
        FloatSquareMatrix
            Matrix of non-randomness parameters.
        """
        return self._c + self._d*(T - 273.15)

    @functools.cache
    def tau(self, T: float) -> FloatSquareMatrix:
        r"""Compute matrix of dimensionless interaction parameters.

        $$ \tau_{ij} = a_{ij} + b_{ij}/T + e_{ij} \ln{T} + f_{ij} T $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.

        Returns
        -------
        FloatSquareMatrix
            Matrix of dimensionless interaction parameters.
        """
        return self._a + self._b/T + self._e*log(T) + self._f*T

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
        tau = self.tau(T)
        alpha = self.alpha(T)
        G = exp(-alpha*tau)
        A = dot(x, tau*G)
        B = dot(x, G)
        return R*T*dot(x, A/B)

    def gamma(self, T: float, x: FloatVector) -> FloatVector:
        r"""Activity coefficients, $\gamma_i$.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector
            Activity coefficients of all components.
        """
        return NRTL_gamma(x, self.tau(T), self.alpha(T))


def NRTL_gamma(x: FloatVector,
               tau: FloatSquareMatrix,
               alpha: FloatSquareMatrix
               ) -> FloatVector:
    r"""Calculate the activity coefficients of a multicomponent mixture
    according to the NRTL model.

    $$ \ln{\gamma_i} = \frac{\displaystyle\sum_{k}{x_{k}\tau_{ki}G_{ki}}}
    {\displaystyle\sum_{k}{x_{k}G_{ki}}}
    +\sum_{j}{\frac{x_{j}G_{ij}}{\displaystyle\sum_{k}{x_{k}G_{kj}}}}
    {\left ({\tau_{ij}-\frac{\displaystyle\sum_{k}{x_{k}\tau_{kj}G_{kj}}}
    {\displaystyle\sum_{k}{x_{k}G_{kj}}}}\right )} $$

    where $x_i$ are the mole fractions, $\tau_{ij}$ are the dimensionless
    interaction parameters, $\alpha_{ij}$ are the non-randomness parameters,
    and $G_{ij}=\exp(-\alpha_{ij} \tau_{ij})$.

    **References**

    *   Renon, H. and Prausnitz, J.M. (1968), Local compositions in
        thermodynamic excess functions for liquid mixtures. AIChE J.,
        14: 135-144.

    Parameters
    ----------
    x : FloatVector
        Mole fractions of all components. Unit = mol/mol.
    tau : FloatSquareMatrix
        Matrix (NxN) of dimensionless interaction parameters, $\tau_{ij}$. It
        is expected (but not checked) that $\tau_{ii}=0$.
    alpha : FloatSquareMatrix
        Matrix (NxN) of non-randomness parameters, $\alpha_{ij}$. It is
        expected (but not checked) that $\alpha_{ij}=\alpha_{ji}$.

    Returns
    -------
    FloatVector
        Activity coefficients of all components.

    !!! note annotate "See also"

        * [`NRTL`](NRTL.md): related class.
    """

    G = exp(-alpha*tau)

    # N = x.size
    # A = np.empty(N)
    # B = np.empty(N)
    # for i in range(N):
    #     A[i] = np.sum(x*tau[:, i]*G[:, i])
    #     B[i] = np.sum(x*G[:, i])
    # C = np.zeros(N)
    # for j in range(N):
    #     C += x[j]*G[:, j]/B[j]*(tau[:, j] - A[j]/B[j])

    A = dot(x, tau*G)
    B = dot(x, G)
    C = dot(tau*G, x/B) - dot(G, x*A/B**2)

    return exp(A/B + C)
