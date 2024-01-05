# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Literal, Union

import numpy as np
from numpy import exp, log

from polykin.utils.tools import (check_bounds, check_type,
                                 convert_check_temperature, custom_repr)
from polykin.utils.types import (FloatOrArray, FloatOrArrayLike, IntOrArray,
                                 IntOrArrayLike)

from .arrhenius import Arrhenius
from .base import KineticCoefficientCLD
from .eyring import Eyring

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

    *   [Smith et al. (2005)](https://doi.org/10.1016/j.eurpolymj.2004.09.002)

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
    """

    kp: Union[Arrhenius, Eyring]
    C: float
    ihalf: float
    symbol: str = 'k_p(i)'

    def __init__(self,
                 kp: Union[Arrhenius, Eyring],
                 C: float = 10.,
                 ihalf: float = 1.0,
                 name: str = ''
                 ) -> None:
        """Construct `PropagationHalfLength` with the given parameters."""

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
    def equation(i: IntOrArray,
                 kp: FloatOrArray,
                 C: float,
                 ihalf: float
                 ) -> FloatOrArray:
        r"""Half-length model chain-length dependence equation.

        Parameters
        ----------
        i : IntOrArray
            Chain length of radical.
        kp : FloatOrArray
            Long-chain value of the propagation rate coefficient, $k_p$.
        C : float
            Ratio of the propagation coefficients of a monomeric radical and a
            long-chain radical, $C$.
        ihalf : float
            Half-length, $i_{i/2}$.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """

        return kp*(1 + (C - 1)*exp(-log(2)*(i - 1)/ihalf))

    def __call__(self,
                 T: FloatOrArrayLike,
                 i: IntOrArrayLike,
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
            Chain length of radical.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """
        TK = convert_check_temperature(T, Tunit, self.kp.Trange)
        if isinstance(i, (list, tuple)):
            i = np.array(i, dtype=np.int32)
        return self.equation(i, self.kp(TK), self.C, self.ihalf)
