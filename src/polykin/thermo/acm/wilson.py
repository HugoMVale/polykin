# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import functools
from typing import Optional

import numpy as np
from numpy import dot, exp, log
from scipy.constants import gas_constant as R

from polykin.utils.exceptions import ShapeError
from polykin.utils.tools import check_bounds
from polykin.utils.types import FloatSquareMatrix, FloatVector

from .base import SmallSpeciesActivityModel

__all__ = ['Wilson', 'Wilson_gamma']


class Wilson(SmallSpeciesActivityModel):
    r"""Wilson multicomponent activity coefficient model.

    This model is based on the following molar excess Gibbs energy
    expression:

    $$ \frac{g^{E}}{RT} =
        -\sum_{i} x_i\ln{\left(\sum_{j}{x_{j}\Lambda_{ij}}\right)} $$

    where $x_i$ are the mole fractions and $\Lambda_{ij}$ are the interaction
    parameters.

    In this particular implementation, the interaction parameters are allowed
    to depend on temperature according to the following empirical relationship
    (as done in Aspen Plus):

    $$ \Lambda_{ij} = \exp( a_{ij} + b_{ij}/T + c_{ij} \ln{T} + d_{ij} T ) $$

    Moreover, $\Lambda_{ij} \neq \Lambda_{ji}$ and $\Lambda_{ii}=1$.

    **References**

    *   G.M. Wilson, J. Am. Chem. Soc., 1964, 86, 127.

    Parameters
    ----------
    N : int
        Number of components.
    a : FloatSquareMatrix (N,N) | None
        Matrix of interaction parameters, by default 0.
    b : FloatSquareMatrix (N,N) | None
        Matrix of interaction parameters, by default 0. Unit = K.
    c : FloatSquareMatrix (N,N) | None
        Matrix of interaction parameters, by default 0.
    d : FloatSquareMatrix (N,N) | None
        Matrix of interaction parameters, by default 0.
    name: str
        Name.

    See also
    --------
    * [`Wilson_gamma`](Wilson_gamma.md): related activity coefficient
    method.

    """

    _a: FloatSquareMatrix
    _b: FloatSquareMatrix
    _c: FloatSquareMatrix
    _d: FloatSquareMatrix

    def __init__(self,
                 N: int,
                 a: Optional[FloatSquareMatrix] = None,
                 b: Optional[FloatSquareMatrix] = None,
                 c: Optional[FloatSquareMatrix] = None,
                 d: Optional[FloatSquareMatrix] = None,
                 name: str = ''
                 ) -> None:

        # Set default values
        if a is None:
            a = np.zeros((N, N))
        if b is None:
            b = np.zeros((N, N))
        if c is None:
            c = np.zeros((N, N))
        if d is None:
            d = np.zeros((N, N))

        # Check shapes
        for array in [a, b, c, d]:
            if array.shape != (N, N):
                raise ShapeError(
                    f"The shape of matrix {array} is invalid: {array.shape}.")

        # Check bounds (same as Aspen Plus)
        check_bounds(a, -50., 50., 'a')
        check_bounds(b, -1.5e4, 1.5e4, 'b')

        # Ensure Lambda_ii=1
        for array in [a, b, c, d]:
            np.fill_diagonal(array, 0.)

        super().__init__(N, name)
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    @functools.cache
    def Lambda(self,
               T: float
               ) -> FloatSquareMatrix:
        r"""Compute the matrix of interaction parameters.

        $$ \Lambda_{ij}=\exp(a_{ij} + b_{ij}/T + c_{ij} \ln{T} + d_{ij} T) $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.

        Returns
        -------
        FloatSquareMatrix (N,N)
            Interaction parameters.
        """
        return exp(self._a + self._b/T + self._c*log(T) + self._d*T)

    def gE(self, T: float, x: FloatVector) -> float:
        return -R*T*dot(x, log(dot(self.Lambda(T), x)))

    def gamma(self, T: float, x: FloatVector) -> FloatVector:
        return Wilson_gamma(x, self.Lambda(T))


def Wilson_gamma(x: FloatVector,
                 Lambda: FloatSquareMatrix
                 ) -> FloatVector:
    r"""Calculate the activity coefficients of a multicomponent mixture
    according to the Wilson model.

    $$ \ln{\gamma_i} = 
    -\ln{\left(\displaystyle\sum_{j}{x_{j}\Lambda_{ij}}\right)} + 1 -
    \sum_{k}{\frac{x_{k}\Lambda_{ki}}
    {\displaystyle\sum_{j}{x_{j}\Lambda_{kj}}}} $$

    where $x_i$ are the mole fractions and $\Lambda_{ij}$ are the interaction
    parameters.

    **References**

    *   G.M. Wilson, J. Am. Chem. Soc., 1964, 86, 127.

    Parameters
    ----------
    x : FloatVector (N)
        Mole fractions of all components. Unit = mol/mol.
    Lambda : FloatSquareMatrix (N,N)
        Interaction parameters, $\Lambda_{ij}$. It is expected (but not
        checked) that $\Lambda_{ii}=1$.

    Returns
    -------
    FloatVector (N)
        Activity coefficients of all components.

    See also
    --------
    * [`Wilson`](Wilson.md): related class.

    """
    A = dot(Lambda, x)
    return exp(1. - dot(x/A, Lambda))/A
