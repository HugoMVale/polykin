# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Literal, Union

import numpy as np
from numpy import sqrt

from polykin.types import (FloatOrArray, FloatOrArrayLike, IntOrArray,
                           IntOrArrayLike)
from polykin.utils import (check_bounds, check_type, convert_check_temperature,
                           custom_repr)

from .arrhenius import Arrhenius
from .base import KineticCoefficientCLD
from .eyring import Eyring

__all__ = ['TerminationCompositeModel']


class TerminationCompositeModel(KineticCoefficientCLD):
    r"""Composite model for the termination rate coefficient between two
    radicals.

    This model implements the chain-length dependence:

    $$ k_t(i,j)=\sqrt{k_t(i,i) k_t(j,j)} $$

    with:

    $$ k_t(i,i)=\begin{cases}
    k_t(1,1)i^{-\alpha_S}& \text{if } i \leq i_{crit} \\
    k_t(1,1)i_{crit}^{-(\alpha_S-\alpha_L)}i^{-\alpha_L} & \text{if }i>i_{crit}
    \end{cases} $$

    where $k_t(1,1)$ is the temperature-dependent termination rate coefficient
    between two monomeric radicals, $i_{crit}$ is the critical chain length,
    $\alpha_S$ is the short-chain exponent, and $\alpha_L$ is the long-chain
    exponent.

    References
    ----------
    *   [Smith & Russel (2003)](https://doi.org/10.1002/mats.200390029)

    Parameters
    ----------
    kt11 : Arrhenius | Eyring
        Temperature-dependent termination rate coefficient between two
        monomeric radicals, $k_t(1,1)$.
    icrit : int
        Critical chain length, $i_{crit}$.
    aS : float
        Short-chain exponent, $\alpha_S$.
    aL : float
        Long-chain exponent, $\alpha_L$.
    name : str
        Name.
    """

    kt11: Union[Arrhenius, Eyring]
    icrit: int
    aS: float
    aL: float
    symbol: str = 'k_t (i,j)'

    def __init__(self,
                 kt11: Union[Arrhenius, Eyring],
                 icrit: int,
                 aS: float = 0.5,
                 aL: float = 0.2,
                 name: str = ''
                 ) -> None:
        """Construct `TerminationCompositeModel` with the given parameters."""

        check_type(kt11, (Arrhenius, Eyring), 'k11')
        check_bounds(icrit, 1, 200, 'icrit')
        check_bounds(aS, 0, 1, 'alpha_short')
        check_bounds(aL, 0, 0.5, 'alpha_long')

        self.kt11 = kt11
        self.icrit = icrit
        self.aS = aS
        self.aL = aL
        self.name = name

    def __repr__(self) -> str:
        return custom_repr(self, ('name', 'icrit', 'aS', 'aL', 'kt11'))

    @staticmethod
    def equation(i: IntOrArray,
                 j: IntOrArray,
                 kt11: FloatOrArray,
                 icrit: int,
                 aS: float,
                 aL: float,
                 ) -> FloatOrArray:
        r"""Composite model chain-length dependence equation.

        Parameters
        ----------
        i : IntOrArray
            Chain length of 1st radical.
        j : IntOrArray
            Chain length of 2nd radical.
        kt11 : FloatOrArray
            Temperature-dependent termination rate coefficient between two
            monomeric radicals, $k_t(1,1)$.
        icrit : int
            Critical chain length, $i_{crit}$.
        aS : float
            Short-chain exponent, $\alpha_S$.
        aL : float
            Long-chain exponent, $\alpha_L$.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """

        def ktii(i):
            return np.where(i <= icrit,
                            kt11*i**(-aS),
                            kt11*icrit**(-aS+aL)*i**(-aL))

        return sqrt(ktii(i)*ktii(j))

    def __call__(self,
                 T: FloatOrArrayLike,
                 i: IntOrArrayLike,
                 j: IntOrArrayLike,
                 Tunit: Literal['C', 'K'] = 'K'
                 ) -> FloatOrArray:
        r"""Evaluate kinetic coefficient at given conditions, including unit
        conversion and range check.

        Parameters
        ----------
        T : FloatOrArrayLike
            Temperature.
            Unit = `Tunit`.
        i : IntOrArrayLike
            Chain length of 1st radical.
        j : IntOrArrayLike
            Chain length of 2nd radical.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """
        TK = convert_check_temperature(T, Tunit, self.kt11.Trange)
        if isinstance(i, (list, tuple)):
            i = np.array(i, dtype=np.int32)
        if isinstance(j, (list, tuple)):
            j = np.array(j, dtype=np.int32)
        return self.equation(i, j, self.kt11(TK), self.icrit, self.aS, self.aL)
