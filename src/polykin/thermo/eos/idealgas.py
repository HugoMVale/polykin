# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Union

import numpy as np
from numpy import log
from scipy.constants import R

from polykin.utils.types import FloatArray, FloatVector

from .base import GasEoS

__all__ = ['IdealGas']


class IdealGas(GasEoS):
    r"""Ideal gas equation of state.

    This EOS is based on the following $P(v,T)$ relationship:

    $$ P = \frac{R T}{v} $$

    where $P$ is the pressure, $T$ is the temperature, and $v$ is the molar
    volume.

    !!! important

        The ideal gas model is very convenient, but its validity is limited to
        low pressures and high temperatures.
    """

    def Z(self, T=None, P=None, y=None):
        r"""Calculate the compressibility factor of the fluid.

        Returns
        -------
        float
            Compressibility factor of the gas.
        """
        return 1.

    def P(self,
          T: Union[float, FloatArray],
          v: Union[float, FloatArray],
          y=None
          ) -> Union[float, FloatArray]:
        r"""Calculate the pressure of the fluid.

        Parameters
        ----------
        T : float | FloatArray
            Temperature. Unit = K.
        v : float | FloatArray
            Molar volume. Unit = m³/mol.

        Returns
        -------
        float | FloatArray
             Pressure. Unit = Pa.
        """
        return R*T/v

    def phi(self, T=None, P=None, y=None) -> FloatVector:
        r"""Calculate the fugacity coefficients of all components in the gas
        phase.

        Returns
        -------
        FloatVector
            Fugacity coefficients of all components.
        """
        if y is None:
            return np.array([1.])
        else:
            return np.ones_like(y)

    def DA(self, T, V, n, v0):
        nt = n.sum()
        return -nt*R*T*log(V/(nt*v0))
