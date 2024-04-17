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
from polykin.utils.types import FloatSquareMatrix, FloatVector, FloatVectorLike

from .base import SmallSpeciesActivityModel

__all__ = ['UNIQUAC', 'UNIQUAC_gamma']


class UNIQUAC(SmallSpeciesActivityModel):
    r"""[UNIQUAC](https://en.wikipedia.org/wiki/UNIQUAC) multicomponent
    activity coefficient model.

    This model is based on the following molar excess Gibbs energy
    expression:

    \begin{aligned}
    g^E &= g^R + g^C \\
    \frac{g^R}{R T} &= -\sum_i q_i x_i \ln{\left ( \sum_j \theta_j \tau_{ji} \right )} \\
    \frac{g^C}{R T} &= \sum_i x_i \ln{\frac{\Phi_i}{x_i}} + 5\sum_i q_ix_i \ln{\frac{\theta_i}{\Phi_i}}
    \end{aligned}

    with:

    \begin{aligned}
    \Phi_i =\frac{x_i r_i}{\sum_j x_j r_j} \\
    \theta_i =\frac{x_i q_i}{\sum_j x_j q_j}
    \end{aligned}

    where $x_i$ are the mole fractions, $q_i$ (a relative surface) and $r_i$
    (a relative volume) denote the pure-component parameters, and $\tau_{ij}$
    are the interaction parameters. 

    In this particular implementation, the interaction parameters are allowed
    to depend on temperature according to the following empirical relationship
    (as done in Aspen Plus):

    $$ \tau_{ij} = \exp( a_{ij} + b_{ij}/T + c_{ij} \ln{T} + d_{ij} T ) $$

    Moreover, $\tau_{ij} \neq \tau_{ji}$ and $\tau_{ii}=1$.

    **References**

    *   Abrams, D.S. and Prausnitz, J.M. (1975), Statistical thermodynamics of
        liquid mixtures: A new expression for the excess Gibbs energy of partly
        or completely miscible systems. AIChE J., 21: 116-128.

    Parameters
    ----------
    N : int
        Number of components.
    q : FloatVectorLike (N)
        Relative surface areas of all components.
    r : FloatVectorLike (N)
        Relative volumes of all components.
    a : FloatSquareMatrix (N,N) | None
        Matrix of interaction parameters, by default 0.
    b : FloatSquareMatrix (N,N) | None
        Matrix of interaction parameters, by default 0. Unit = K.
    c : FloatSquareMatrix (N,N) | None
        Matrix of interaction parameters, by default 0.
    d : FloatSquareMatrix (N,N) | None
        Matrix of interaction parameters, by default 0.
    name : str
        Name.

    See also
    --------
    * [`UNIQUAC_gamma`](UNIQUAC_gamma.md): related activity coefficient method.

    """

    _q: FloatVector
    _r: FloatVector
    _a: FloatSquareMatrix
    _b: FloatSquareMatrix
    _c: FloatSquareMatrix
    _d: FloatSquareMatrix

    def __init__(self,
                 N: int,
                 q: FloatVectorLike,
                 r: FloatVectorLike,
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

        # Check shapes -> move to func
        q = np.asarray(q)
        r = np.asarray(r)
        for vector in [q, r]:
            if vector.shape != (N,):
                raise ShapeError(
                    f"The shape of vector {vector} is invalid: {vector.shape}.")
        for array in [a, b, c, d]:
            if array.shape != (N, N):
                raise ShapeError(
                    f"The shape of matrix {array} is invalid: {array.shape}.")

        # Check bounds (same as Aspen Plus)
        check_bounds(a, -50., 50., 'a')
        check_bounds(b, -1.5e4, 1.5e4, 'b')

        # Ensure tau_ii=1
        for array in [a, b, c, d]:
            np.fill_diagonal(array, 0.)

        super().__init__(N, name)
        self._q = q
        self._r = r
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    @functools.cache
    def tau(self,
            T: float
            ) -> FloatSquareMatrix:
        r"""Compute the matrix of interaction parameters.

        $$ \tau_{ij} = \exp( a_{ij} + b_{ij}/T + c_{ij} \ln{T} + d_{ij} T ) $$

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

    def gE(self,
           T: float,
           x: FloatVector
           ) -> float:

        r = self._r
        q = self._q
        tau = self.tau(T)

        phi = x*r/dot(x, r)
        theta = x*q/dot(x, q)

        p = x > 0.
        gC = np.sum(x[p]*(log(phi[p]/x[p]) + 5*q[p]*log(theta[p]/phi[p])))
        gR = -np.sum(q[p]*x[p]*log(dot(theta, tau)[p]))

        return R*T*(gC + gR)

    def gamma(self,
              T: float,
              x: FloatVector
              ) -> FloatVector:
        return UNIQUAC_gamma(x, self._q, self._r, self.tau(T))


def UNIQUAC_gamma(x: FloatVector,
                  q: FloatVector,
                  r: FloatVector,
                  tau: FloatSquareMatrix
                  ) -> FloatVector:
    r"""Calculate the activity coefficients of a multicomponent mixture
    according to the UNIQUAC model.

    \begin{aligned}
    \ln{\gamma_i} &= \ln{\gamma_i^R} + \ln{\gamma_i^C} \\
    \ln{\gamma_i^R} &= q_i \left(1 - \ln{s_i} - 
                       \sum_j \theta_j\frac{\tau_{ij}}{s_j} \right) \\
    \ln{\gamma_i^C} &= 1 - J_i + \ln{J_i} - 5 q_i \left(1 - \frac{J_i}{L_i} +
                      \ln{\frac{J_i}{L_i}} \right)
    \end{aligned}

    with:

    \begin{aligned}
    J_i &=\frac{r_i}{\sum_j x_j r_j} \\
    L_i &=\frac{q_i}{\sum_j x_j q_j} \\
    \theta_i &= x_i L_i \\
    s_i &= \sum_j \theta_j \tau_{ji}
    \end{aligned}

    where $x_i$ are the mole fractions, $q_i$ and $r_i$ denote the
    pure-component parameters, and $\tau_{ij}$ are the interaction parameters.

    **References**

    *   Abrams, D.S. and Prausnitz, J.M. (1975), Statistical thermodynamics of
        liquid mixtures: A new expression for the excess Gibbs energy of partly
        or completely miscible systems. AIChE J., 21: 116-128.
    *   JM Smith, HC Van Ness, MM Abbott. Introduction to chemical engineering
        thermodynamics, 5th edition, 1996, p. 740-741.

    Parameters
    ----------
    x : FloatVector (N)
        Mole fractions of all components. Unit = mol/mol.
    q : FloatVector (N)
        Relative surface areas of all components.
    r : FloatVector (N)
        Relative volumes of all components. 
    tau : FloatSquareMatrix (N,N)
        Interaction parameters, $\tau_{ij}$. It is expected (but not checked)
        that $\tau_{ii}=1$.

    Returns
    -------
    FloatVector (N)
        Activity coefficients of all components.

    See also
    --------
    * [`UNIQUAC`](UNIQUAC.md): related class.

    """

    J = r/dot(x, r)
    L = q/dot(x, q)
    theta = x*L
    s = dot(theta, tau)

    return J*exp(1 - J + q*(1 - log(s) - dot(tau, theta/s)
                            - 5*(1 - J/L + log(J/L))))
