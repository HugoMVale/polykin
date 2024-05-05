# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Literal, Union

import numpy as np
from numpy import exp, log

from polykin.kinetics.arrhenius import Arrhenius
from polykin.kinetics.base import KineticCoefficientCLD
from polykin.kinetics.eyring import Eyring
from polykin.utils.tools import (check_bounds, check_type,
                                 convert_check_temperature, custom_repr)
from polykin.utils.types import (FloatArray, FloatArrayLike, IntArray,
                                 IntArrayLike)

__all__ = ['PropagationHalfLength']


class PropagationHalfLength(KineticCoefficientCLD):
    r"""Half-length model for the decay of the propagation rate coefficient
    with chain length.

    This model implements the chain-length dependence:

    $$ k_p(i) = k_p \left[ 1+ (C - 1)\exp{\left (-\frac{\ln 2}{i_{1/2}} (i-1)
                \right )} \right] $$

    where $k_p=k_p(\infty)$ is the long-chain value of the propagation rate
    coefficient, $C\ge 1$ is the ratio $k_p(1)/k_p$ and $(i_{1/2}+1)$ is the
    hypothetical chain-length at which the difference $k_p(1) - k_p$ is halved.

    **References**

    *   Smith, Gregory B., et al. "The effects of chain length dependent
        propagation and termination on the kinetics of free-radical
        polymerization at low chain lengths." European polymer journal
        41.2 (2005): 225-230.

    Parameters
    ----------
    kp : Arrhenius | Eyring
        Long-chain value of the propagation rate coefficient, $k_p$.
    C : float
        Ratio of the propagation coefficients of a monomeric radical and a
        long-chain radical, $C$.
    ihalf : float
        Half-length, $i_{1/2}$.
    name : str
        Name.

    Examples
    --------
    >>> from polykin.kinetics import PropagationHalfLength, Arrhenius
    >>> kp = Arrhenius(
    ...    10**7.63, 32.5e3/8.314, Tmin=261., Tmax=366.,
    ...    symbol='k_p', unit='L/mol/s', name='kp of styrene')

    >>> kpi = PropagationHalfLength(kp, C=10, ihalf=0.5,
    ...    name='kp(T,i) of styrene')
    >>> kpi
    name:    kp(T,i) of styrene
    C:       10
    ihalf:   0.5
    kp:
      name:            kp of styrene
      symbol:          k_p
      unit:            L/mol/s
      Trange [K]:      (261.0, 366.0)
      k0 [L/mol/s]:    42657951.88015926
      EaR [K]:         3909.0690401732018
      T0 [K]:          inf

    >>> kpi(T=50., i=3, Tunit='C')
    371.75986615653215
    """

    kp: Union[Arrhenius, Eyring]
    C: float
    ihalf: float
    symbol: str = 'k_p(i)'

    def __init__(self,
                 kp: Union[Arrhenius, Eyring],
                 C: float = 10.0,
                 ihalf: float = 1.0,
                 name: str = ''
                 ) -> None:

        check_type(kp, (Arrhenius, Eyring), 'kp')
        check_bounds(C, 1., 100., 'C')
        check_bounds(ihalf, 0.1, 10, 'ihalf')

        self.kp = kp
        self.C = C
        self.ihalf = ihalf
        self.name = name

    def __repr__(self) -> str:
        return custom_repr(self, ('name', 'C', 'ihalf', 'kp'))

    @staticmethod
    def equation(i: Union[int, IntArray],
                 kp: Union[float, FloatArray],
                 C: float,
                 ihalf: float
                 ) -> Union[float, FloatArray]:
        r"""Half-length model chain-length dependence equation.

        Parameters
        ----------
        i : int | IntArray
            Chain length of radical.
        kp : float | FloatArray
            Long-chain value of the propagation rate coefficient, $k_p$.
        C : float
            Ratio of the propagation coefficients of a monomeric radical and a
            long-chain radical, $C$.
        ihalf : float
            Half-length, $i_{i/2}$.

        Returns
        -------
        float | FloatArray
            Coefficient value.
        """

        return kp*(1 + (C - 1)*exp(-log(2)*(i - 1)/ihalf))

    def __call__(self,
                 T: Union[float, FloatArrayLike],
                 i: Union[int, IntArrayLike],
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
            Chain length of radical.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        float | FloatArray
            Coefficient value.
        """
        TK = convert_check_temperature(T, Tunit, self.kp.Trange)
        if isinstance(i, (list, tuple)):
            i = np.array(i, dtype=np.int32)
        return self.equation(i, self.kp(TK), self.C, self.ihalf)
