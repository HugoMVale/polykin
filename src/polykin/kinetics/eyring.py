# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from __future__ import annotations

import numpy as np
from numpy import exp
from scipy.constants import Boltzmann as kB
from scipy.constants import R, h

from polykin.math import convert_FloatOrArrayLike_to_FloatOrArray
from polykin.types import FloatOrArray, FloatOrArrayLike
from polykin.utils import check_bounds, check_shapes

from .base import KineticCoefficientT

__all__ = ['Eyring']


class Eyring(KineticCoefficientT):
    r"""[Eyring](https://en.wikipedia.org/wiki/Eyring_equation) kinetic rate
    coefficient.

    This class implements the following temperature dependence:

    $$
    k(T) = \dfrac{\kappa k_B T}{h}
           \exp\left(\frac{\Delta S^\ddagger}{R}\right)
           \exp\left(-\frac{\Delta H^\ddagger}{R T}\right)
    $$

    where $\kappa$ is the transmission coefficient, $\Delta S^\ddagger$ is
    the entropy of activation, and $\Delta H^\ddagger$ is the enthalpy of
    activation. The unit of $k$ is physically set to s$^{-1}$.

    Parameters
    ----------
    DSa : FloatOrArrayLike
        Entropy of activation, $\Delta S^\ddagger$.
        Unit = J/(mol·K).
    DHa : FloatOrArrayLike
        Enthalpy of activation, $\Delta H^\ddagger$.
        Unit = J/mol.
    kappa : FloatOrArrayLike
        Transmission coefficient.
    Tmin : FloatOrArrayLike
        Lower temperature bound.
        Unit = K.
    Tmax : FloatOrArrayLike
        Upper temperature bound.
        Unit = K.
    symbol : str
        Symbol of coefficient $k$.
    name : str
        Name.

    !!! note annotate "See also"

        * [`Arrhenius`](Arrhenius.md): alternative method.

    """

    _pinfo = {'DSa': ('J/(mol·K)', True),
              'DHa': ('J/mol', True), 'kappa': ('', False)}

    def __init__(self,
                 DSa: FloatOrArrayLike,
                 DHa: FloatOrArrayLike,
                 kappa: FloatOrArrayLike = 1.0,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = np.inf,
                 symbol: str = 'k',
                 name: str = ''
                 ) -> None:
        """Construct `Eyring` with the given parameters."""

        # Convert lists to arrays
        DSa, DHa, kappa, Tmin, Tmax = \
            convert_FloatOrArrayLike_to_FloatOrArray(
                [DSa, DHa, kappa, Tmin, Tmax])

        # Check shapes
        self._shape = check_shapes([DSa, DHa], [kappa, Tmin, Tmax])

        # Check bounds
        check_bounds(DSa, 0., np.inf, 'DSa')
        check_bounds(DHa, 0., np.inf, 'DHa')
        check_bounds(kappa, 0., 1., 'kappa')

        self.p = {'DSa': DSa, 'DHa': DHa, 'kappa': kappa}
        super().__init__((Tmin, Tmax), '1/s', symbol, name)

    @staticmethod
    def equation(T: FloatOrArray,
                 DSa: FloatOrArray,
                 DHa: FloatOrArray,
                 kappa: FloatOrArray,
                 ) -> FloatOrArray:
        r"""Eyring equation.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        DSa : FloatOrArray
            Entropy of activation, $\Delta S^\ddagger$.
            Unit = J/(mol·K).
        DHa : FloatOrArray
            Enthalpy of activation, $\Delta H^\ddagger$.
            Unit = J/mol.
        kappa : FloatOrArray
            Transmission coefficient.

        Returns
        -------
        FloatOrArray
            Coefficient value. Unit = 1/s.
        """
        return kappa * kB*T/h * exp((DSa - DHa/T)/R)
