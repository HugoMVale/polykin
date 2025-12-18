# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from __future__ import annotations

import numpy as np
from numpy import exp
from scipy.constants import Boltzmann as kB
from scipy.constants import R, h

from polykin.kinetics.coefficients.base import KineticCoefficientT
from polykin.utils.math import convert_FloatOrArrayLike_to_FloatOrArray
from polykin.utils.tools import check_bounds, check_shapes
from polykin.utils.types import FloatArray, FloatArrayLike

__all__ = ["Eyring"]


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
    DSa : float | FloatArrayLike
        Entropy of activation, $\Delta S^\ddagger$ [J/(mol·K)].
    DHa : float | FloatArrayLike
        Enthalpy of activation, $\Delta H^\ddagger$ [J/mol].
    kappa : float | FloatArrayLike
        Transmission coefficient.
    Tmin : float | FloatArrayLike
        Lower temperature bound [K].
    Tmax : float | FloatArrayLike
        Upper temperature bound [K].
    symbol : str
        Symbol of coefficient $k$.
    name : str
        Name.

    See Also
    --------
    * [`Arrhenius`](Arrhenius.md): alternative method.

    Examples
    --------
    Define and evaluate a rate coefficient from transition state properties.
    >>> from polykin.kinetics import Eyring
    >>> k = Eyring(
    ...     DSa=20.,
    ...     DHa=5e4,
    ...     kappa=0.8,
    ...     Tmin=273., Tmax=373.,
    ...     symbol='k',
    ...     name='A->B')
    >>> k(25.,'C')
    95808.36742009166
    """

    _pinfo = {"DSa": ("J/(mol·K)", True), "DHa": ("J/mol", True), "kappa": ("", False)}

    def __init__(
        self,
        DSa: float | FloatArrayLike,
        DHa: float | FloatArrayLike,
        kappa: float | FloatArrayLike = 1.0,
        Tmin: float | FloatArrayLike = 0.0,
        Tmax: float | FloatArrayLike = np.inf,
        symbol: str = "k",
        name: str = "",
    ) -> None:

        # Convert lists to arrays
        DSa, DHa, kappa, Tmin, Tmax = convert_FloatOrArrayLike_to_FloatOrArray(
            [DSa, DHa, kappa, Tmin, Tmax]
        )

        # Check shapes
        self._shape = check_shapes([DSa, DHa], [kappa, Tmin, Tmax])

        # Check bounds
        check_bounds(DSa, 0.0, np.inf, "DSa")
        check_bounds(DHa, 0.0, np.inf, "DHa")
        check_bounds(kappa, 0.0, 1.0, "kappa")

        self.p = {"DSa": DSa, "DHa": DHa, "kappa": kappa}
        super().__init__((Tmin, Tmax), "1/s", symbol, name)

    @staticmethod
    def equation(
        T: float | FloatArray,
        DSa: float | FloatArray,
        DHa: float | FloatArray,
        kappa: float | FloatArray,
    ) -> float | FloatArray:
        r"""Eyring equation.

        Parameters
        ----------
        T : float | FloatArray
            Temperature [K].
        DSa : float | FloatArray
            Entropy of activation, $\Delta S^\ddagger$ [J/(mol·K)].
        DHa : float | FloatArray
            Enthalpy of activation, $\Delta H^\ddagger$ [J/mol].
        kappa : float | FloatArray
            Transmission coefficient.

        Returns
        -------
        float | FloatArray
            Coefficient value [s⁻¹].
        """
        return kappa * kB * T / h * exp((DSa - DHa / T) / R)
