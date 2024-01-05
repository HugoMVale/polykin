# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from numpy import log
from scipy.constants import R

from polykin.utils.types import FloatOrArray

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
          T: FloatOrArray,
          v: FloatOrArray,
          y=None
          ) -> FloatOrArray:
        r"""Calculate the pressure of the fluid.

        Parameters
        ----------
        T : FloatOrArray
            Temperature. Unit = K.
        v : FloatOrArray
            Molar volume. Unit = m³/mol.

        Returns
        -------
        FloatOrArray
             Pressure. Unit = Pa.
        """
        return R*T/v

    def phi(self, T=None, P=None, y=None):
        r"""Calculate the fugacity coefficients of all components in the gas
        phase.

        Returns
        -------
        FloatVector
            Fugacity coefficients of all components.
        """
        return 1.

    def DA(self, T, V, n, v0):
        nt = n.sum()
        return -nt*R*T*log(V/(nt*v0))
