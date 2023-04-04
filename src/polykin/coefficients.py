# %% Coefficients

from polykin.utils import check_bounds, FloatOrArray, FloatOrArrayLike, \
    IntOrArrayLike
from polykin.base import Base
from scipy.constants import h, R, Boltzmann as kB

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Union


class Coefficient(Base, ABC):
    """Abstract coefficient."""
    pass


class CoefficientT(Coefficient):
    """Abstract temperature dependent coefficient."""

    def __init__(self):
        self._shape = None

    def __call__(self,
                 T: FloatOrArrayLike,
                 kelvin: bool = False
                 ) -> FloatOrArray:
        r"""Evaluate coefficient at desired temperature.

        Parameters
        ----------
        T : float | list[float] | ndarray
            Temperature (default °C).
        kelvin : bool, optional
            Switch temperature unit between °C (if `False`) and K (if `True`).

        Returns
        -------
        float | ndarray
            Rate coefficient, k(T).
        """
        if isinstance(T, list):
            T = np.asarray(T, dtype=np.float64)
        if kelvin:
            TK = T
        else:
            TK = T + 273.15
        if np.any(TK < 0):  # adds 5 us
            raise ValueError("Invalid `T` input.")
        return self.eval(TK)

    @abstractmethod
    def eval(self, T: FloatOrArray) -> FloatOrArray:
        pass

    @property
    def shape(self):
        return self._shape

    @staticmethod
    def _check_shapes(a: list, b: list) -> Union[None, Any]:
        """Check shape homogeneity between objects in lists `a` and `b`.

        Rules:
        - All objects in `a` must have the same shape, i.e. either all floats
        or all arrays with same shape.
        - Objects in `b` that are arrays, must have identical shape to the
        objects in `a`.
        """
        check_a = True
        check_b = True
        shape = None
        shapes_a = [elem.shape for elem in a if isinstance(elem, np.ndarray)]
        shapes_b = [elem.shape for elem in b if isinstance(elem, np.ndarray)]
        if shapes_a:
            if len(shapes_a) != len(a) or len(set(shapes_a)) != 1:
                check_a = False
            else:
                shape = shapes_a[0]
        if shapes_b:
            if len(set(shapes_a + shapes_b)) != 1:
                check_b = False

        if not (check_a and check_b):
            raise ValueError("Inputs have inconsistent shapes.")

        return shape


class Arrhenius(CoefficientT):
    r"""[Arrhenius](https://en.wikipedia.org/wiki/Arrhenius_equation)
    coefficient, with temperature dependence given by:

    $$ k(T)=k_0\exp\left(-\frac{E_a}{R}\left(\frac{1}{T}-\frac{1}{T_0} \\
        \right)\right) $$

    where $T_0$ is a convenient reference temperature, $E_a$ is the activation
    energy, and $k_0=k(T_0)$. In the limit $T\rightarrow+\infty$, we recover
    the usual form of Arrhenius equation with $k_0=A$.
    """

    def __init__(self,
                 k0: FloatOrArrayLike,
                 EaR: FloatOrArrayLike,
                 T0: FloatOrArrayLike = np.inf,
                 name: str = ''):

        # convert lists to arrays
        if isinstance(k0, list):
            k0 = np.asarray(k0, dtype=np.float64)
        if isinstance(EaR, list):
            EaR = np.asarray(EaR, dtype=np.float64)
        if isinstance(T0, list):
            T0 = np.asarray(T0, dtype=np.float64)

        # check shapes
        self._shape = self._check_shapes([k0, EaR], [T0])

        # check bounds
        check_bounds(k0, 0, np.inf, 'k0')
        check_bounds(EaR, 0, np.inf, 'EaR')
        check_bounds(T0, 0, np.inf, 'T0')

        self.k0 = k0
        self.EaR = EaR
        self.T0 = T0
        self.name = name

    def eval(self, T):
        return self.k0*np.exp(-self.EaR*(1/T - 1/self.T0))


class Eyring(CoefficientT):
    r"""[Eyring](https://en.wikipedia.org/wiki/Eyring_equation) coefficient,
    with temperature dependence given by:

    $$ k(T)=\dfrac{\kappa k_B T}{h} \exp\left(\frac{\Delta S^\ddagger}{R}\right) \\
        \exp\left(-\frac{\Delta H^\ddagger}{R T}\right)$$

    where $\kappa$ is the transmission coefficient, $\Delta S^\ddagger$ is
    the entropy of activation, and $\Delta H^\ddagger$ is the enthalpy of
    activation.
    """

    def __init__(self,
                 DSa: FloatOrArrayLike,
                 DHa: FloatOrArrayLike,
                 kappa: FloatOrArrayLike = 1,
                 name: str = ''):

        # convert lists to arrays
        if isinstance(DSa, list):
            DSa = np.asarray(DSa, dtype=np.float64)
        if isinstance(DHa, list):
            DHa = np.asarray(DHa, dtype=np.float64)
        if isinstance(kappa, list):
            kappa = np.asarray(kappa, dtype=np.float64)

        # check shapes
        self._shape = self._check_shapes([DSa, DHa], [kappa])

        # check bounds
        check_bounds(DSa, 0, np.inf, 'DSa')
        check_bounds(DHa, 0, np.inf, 'DHa')
        check_bounds(kappa, 0, 1, 'kappa')

        self.DSa = DSa
        self.DHa = DHa
        self.kappa = kappa
        self.name = name

    def eval(self, T):
        return self.kappa*kB*T/h*np.exp((self.DSa-self.DHa/T)/R)


class CoefficientCLD(Coefficient):
    """Abstract chain-length dependent (CLD) coefficient."""
    pass


class TerminationCompositeModel(CoefficientCLD):
    r"""Composite model for the termination rate coefficient, with chain-length
    dependence given by:

    $$ k_t(i,j)=\sqrt{k_t(i,i) k_t(j,j)} $$

    with:

    $$ k_t(i,i)=\begin{cases}
    k_t(1,1)i^{-\alpha_S}& \text{ if } i \leq i_{crit} \\
    k_t(1,1)i_{crit}^{-(\alpha_S-\alpha_L)} i^{-\alpha_L} & \text{ if } i > i_{crit}
    \end{cases}
    $$

    where $k_t(1,1)$ is the temperature-dependent termination rate coefficient
    between two monomeric radicals, $i_{crit}$ is the critical chain length, 
    $\alpha_S$ is the short-chain exponent, and $\alpha_L$ is the long-chain
    exponent.
    """

    def __init__(self,
                 k11: CoefficientT,
                 icrit: int,
                 alpha_short: float,
                 alpha_long: float,
                 name: str = ''):

        check_bounds(icrit, 1, 200, 'icrit')
        check_bounds(alpha_short, 0, 1, 'alpha_short')
        check_bounds(alpha_long, 0, 0.5, 'alpha_long')

        self.k11 = k11
        self.icrit = icrit
        self.alpha_short = alpha_short
        self.alpha_long = alpha_long
        self.name = name

    def eval(self,
             T: FloatOrArray,
             i: IntOrArrayLike,
             j: IntOrArrayLike
             ) -> FloatOrArray:

        k11 = self.k11.eval(T)
        aS = self.alpha_short
        aL = self.alpha_long
        icrit = self.icrit

        def ktii(i):
            return np.where(i <= icrit,
                            k11*i**(-aS),
                            k11*icrit**(-aS+aL)*i**(-aL))

        return np.sqrt(ktii(i)*ktii(j))


class DIPPR(CoefficientT):
    """[DIPPR]() temperature dependent coefficient."""
    pass
