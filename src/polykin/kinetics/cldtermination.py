# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Literal, Union

import numpy as np
from numpy import sqrt

from polykin.kinetics.arrhenius import Arrhenius
from polykin.kinetics.base import KineticCoefficientCLD
from polykin.kinetics.eyring import Eyring
from polykin.utils.tools import (check_bounds, check_type,
                                 convert_check_temperature, custom_repr)
from polykin.utils.types import (FloatArray, FloatArrayLike, IntArray,
                                 IntArrayLike)

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

    **References**

    *   Smith, Gregory B., Gregory T. Russell, and Johan PA Heuts. "Termination
        in dilute-solution free-radical polymerization: a composite model."
        Macromolecular theory and simulations 12.5 (2003): 299-314.

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

    Examples
    --------
    >>> from polykin.kinetics import TerminationCompositeModel, Arrhenius
    >>> kt11 = Arrhenius(1e9, 2e3, T0=298.,
    ...        symbol='k_t(T,1,1)', unit='L/mol/s', name='kt11 of Y')

    >>> ktij = TerminationCompositeModel(kt11, icrit=30, name='ktij of Y')
    >>> ktij
    name:    ktij of Y
    icrit:   30
    aS:      0.5
    aL:      0.2
    kt11:
      name:            kt11 of Y
      symbol:          k_t(T,1,1)
      unit:            L/mol/s
      Trange [K]:      (0.0, inf)
      k0 [L/mol/s]:    1000000000.0
      EaR [K]:         2000.0
      T0 [K]:          298.0

    >>> ktij(T=25., i=150, j=200, Tunit='C')
    129008375.03821689
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
    def equation(i: Union[int, IntArray],
                 j: Union[int, IntArray],
                 kt11: Union[float, FloatArray],
                 icrit: int,
                 aS: float,
                 aL: float,
                 ) -> Union[float, FloatArray]:
        r"""Composite model chain-length dependence equation.

        Parameters
        ----------
        i : int | IntArray
            Chain length of 1st radical.
        j : int | IntArray
            Chain length of 2nd radical.
        kt11 : float | FloatArray
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
        float | FloatArray
            Coefficient value.
        """

        def ktii(i):
            return np.where(i <= icrit,
                            kt11*i**(-aS),
                            kt11*icrit**(-aS+aL)*i**(-aL))

        return sqrt(ktii(i)*ktii(j))

    def __call__(self,
                 T: Union[float, FloatArrayLike],
                 i: Union[int, IntArrayLike],
                 j: Union[int, IntArrayLike],
                 Tunit: Literal['C', 'K'] = 'K'
                 ) -> Union[float, FloatArray]:
        r"""Evaluate kinetic coefficient at given conditions, including unit
        conversion and range check.

        Parameters
        ----------
        T : float | FloatArrayLike
            Temperature.
            Unit = `Tunit`.
        i : int | IntArrayLike
            Chain length of 1st radical.
        j : int | IntArrayLike
            Chain length of 2nd radical.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        float | FloatArray
            Coefficient value.
        """
        TK = convert_check_temperature(T, Tunit, self.kt11.Trange)
        if isinstance(i, (list, tuple)):
            i = np.array(i, dtype=np.int32)
        if isinstance(j, (list, tuple)):
            j = np.array(j, dtype=np.int32)
        return self.equation(i, j, self.kt11(TK), self.icrit, self.aS, self.aL)
