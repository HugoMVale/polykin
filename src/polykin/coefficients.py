# %% Coefficients

from polykin.utils import check_bounds, FloatOrArray, FloatOrArrayLike, \
    IntOrArrayLike, eps, RangeWarning, RangeError, ShapeError, FloatRange
from polykin.base import Base

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, R, Boltzmann as kB
from abc import ABC, abstractmethod
from typing import Union, Literal
from warnings import warn


class Coefficient(Base, ABC):
    """_Abstract_ class for all coefficients, c(...)."""

    def __init__(self) -> None:
        self._shape = None

    @property
    def shape(self) -> Union[tuple[int, ...], None]:
        """Shape of underlying coefficient array."""
        return self._shape

    @staticmethod
    def _check_shapes(a: list, b: list) -> Union[tuple[int, ...], None]:
        """Check shape homogeneity between objects in lists `a` and `b`.

        Rules:
        - All objects in `a` must have the same shape, i.e., either all floats
        or all arrays with same shape.
        - Objects in `b` that are arrays, must have identical shape to the
        objects in `a`.

        Parameters
        ----------
        a : list
            List of objects which must have the same shape.
        b : list
            List of objects which, if arrays, must have identical shape to the
            objects in `a`.

        Returns
        -------
        Union[tuple[int, ...], None]
            Common shape of `a` or None.
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
            raise ShapeError("Input parameters have inconsistent shapes.")
        return shape

    @abstractmethod
    def __call__(self, *args) -> FloatOrArray:
        pass

    @abstractmethod
    def eval(self, *args) -> FloatOrArray:
        pass


class CoefficientX1(Coefficient):
    r"""_Abstract_ class for 1-variable coefficient, $c(x)$.
    """
    pass


class CoefficientX2(Coefficient):
    r"""_Abstract_ class for 2-variable coefficient, $c(x, y)$.
    """
    pass


class CoefficientT(CoefficientX1):
    """_Abstract_ temperature-dependent coefficient."""

    Tmin: FloatOrArray
    Tmax: FloatOrArray

    def __call__(self,
                 T: FloatOrArrayLike,
                 Tunit: Literal['C', 'K'] = 'C'
                 ) -> FloatOrArray:
        r"""Evaluate coefficient at given temperature, including unit
        conversion and range check.

        Parameters
        ----------
        T : FloatOrArrayLike
            Temperature.
            Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """
        TK = self._convert_check_temperature(T, Tunit)
        return self.eval(TK)

    @abstractmethod
    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate coefficient at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """
        pass

    def _convert_check_temperature(self,
                                   T: FloatOrArrayLike,
                                   Tunit: Literal['C', 'K']
                                   ) -> FloatOrArray:
        if isinstance(T, list):
            T = np.array(T, dtype=np.float64)
        if Tunit == 'K':
            TK = T
        elif Tunit == 'C':
            TK = T + 273.15
        else:
            raise ValueError("Invalid `Tunit` input.")
        if np.any(TK < 0):
            raise RangeError("`T` must be > 0 K.")
        if np.any(TK < self.Tmin) or np.any(TK > self.Tmax):
            warn("`T` input is outside validity range [Tmin, Tmax].",
                 RangeWarning)
        return TK

    def plot(self,
             kind: Literal['default', 'Arrhenius'] = 'default',
             Trange: FloatRange = [],
             Tunit: Literal['C', 'K'] = 'K',
             title: Union[str, None] = None,
             ) -> None:

        # x-axis vector
        if not (len(Trange) == 2 and Trange[1] > Trange[0]):
            if self._shape:
                Trange = (np.min(self.Tmin), np.max(self.Tmax))
            else:
                Trange = (self.Tmin, self.Tmax)
        x = np.linspace(*Trange, 100)

        fig, ax = plt.subplots()
        if title is None:
            title = f"Coefficient: {self.name}"
        fig.suptitle(title)

        y = self.__call__(x, Tunit)

        if kind == 'default':
            ax.plot(x, y)
        elif kind == 'Arrhenius':
            ax.plot(1/x, np.log(y))

        return None


class Arrhenius(CoefficientT):
    r"""[Arrhenius](https://en.wikipedia.org/wiki/Arrhenius_equation) kinetic
    rate coefficient.

    The temperature dependence is given by:

    $$ k(T)=k_0\exp\left(-\frac{E_a}{R}\left(\frac{1}{T}-\frac{1}{T_0} \\
        \right)\right) $$

    where $T_0$ is a convenient reference temperature, $E_a$ is the activation
    energy, and $k_0=k(T_0)$. In the limit $T\rightarrow+\infty$, the usual
    form of Arrhenius equation with $k_0=A$ is recovered.

    Parameters
    ----------
    k0 : FloatOrArrayLike
        Rate coefficient at the reference temperature, $k_0=k(T_0)$.
        Unit = Any.
    EaR : FloatOrArrayLike
        Energy of activation, $E_a/R$.
        Unit = K.
    T0 : FloatOrArrayLike
        Reference temperature, $T_0$.
        Unit = K.
    Tmin : FloatOrArrayLike
        Lower temperature bound.
        Unit = K.
    Tmax : FloatOrArrayLike
        Upper temperature bound.
        Unit = K.
    name : str
        Name.
    """

    def __init__(self,
                 k0: FloatOrArrayLike,
                 EaR: FloatOrArrayLike,
                 T0: FloatOrArrayLike = np.inf,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = +np.inf,
                 name: str = ''
                 ) -> None:

        # convert lists to arrays
        if isinstance(k0, list):
            k0 = np.array(k0, dtype=np.float64)
        if isinstance(EaR, list):
            EaR = np.array(EaR, dtype=np.float64)
        if isinstance(T0, list):
            T0 = np.array(T0, dtype=np.float64)
        if isinstance(Tmin, list):
            Tmin = np.array(Tmin, dtype=np.float64)
        if isinstance(Tmax, list):
            Tmax = np.array(Tmax, dtype=np.float64)

        # check shapes
        self._shape = self._check_shapes([k0, EaR], [T0, Tmin, Tmax])

        # check bounds
        check_bounds(k0, 0, np.inf, 'k0')
        check_bounds(EaR, 0, np.inf, 'EaR')
        check_bounds(T0, 0, np.inf, 'T0')
        check_bounds(Tmin, 0, np.inf, 'Tmin')
        check_bounds(Tmax, 0, np.inf, 'Tmax')
        check_bounds(Tmax-Tmin, eps, np.inf, 'Tmax-Tmin')

        self.k0 = k0
        self.EaR = EaR
        self.T0 = T0
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.name = name

    def __mul__(self, other):
        """Multipy Arrhenius coefficients.

        Create new Arrhenius coefficient from the product of two Arrhenius
        coefficients.

        Parameters
        ----------
        other : Arrhenius
            Another coefficient.

        Returns
        -------
        Arrhenius
            Product coefficient.
        """
        if isinstance(other, Arrhenius):
            if self._shape == other._shape:
                return Arrhenius(k0=self.eval(np.inf)*other.eval(np.inf),
                                 EaR=self.EaR + other.EaR,
                                 Tmin=np.maximum(self.Tmin, other.Tmin),
                                 Tmax=np.minimum(self.Tmax, other.Tmax),
                                 name=f"{self.name}*{other.name}")
            else:
                raise ShapeError(
                    "Product of coefficients requires identical shapes.")
        else:
            return NotImplemented

    @property
    def A(self) -> FloatOrArray:
        r"""Pre-exponential factor, $A$"""
        return self.eval(np.inf)

    def eval(self, T):
        return self.k0*np.exp(-self.EaR*(1/T - 1/self.T0))


class Eyring(CoefficientT):
    r"""[Eyring](https://en.wikipedia.org/wiki/Eyring_equation) kinetic rate
    coefficient.

    The temperature dependence is given by:

    $$ k(T)=\dfrac{\kappa k_B T}{h} \\
        \exp\left(\frac{\Delta S^\ddagger}{R}\right) \\
        \exp\left(-\frac{\Delta H^\ddagger}{R T}\right)$$

    where $\kappa$ is the transmission coefficient, $\Delta S^\ddagger$ is
    the entropy of activation, and $\Delta H^\ddagger$ is the enthalpy of
    activation.

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
        Unit = dimensionless.
    Tmin : FloatOrArrayLike
        Lower temperature bound.
        Unit = K.
    Tmax : FloatOrArrayLike
        Upper temperature bound.
        Unit = K.
    name : str
        Name.
    """

    def __init__(self,
                 DSa: FloatOrArrayLike,
                 DHa: FloatOrArrayLike,
                 kappa: FloatOrArrayLike = 1.0,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = +np.inf,
                 name: str = ''
                 ) -> None:

        # convert lists to arrays
        if isinstance(DSa, list):
            DSa = np.array(DSa, dtype=np.float64)
        if isinstance(DHa, list):
            DHa = np.array(DHa, dtype=np.float64)
        if isinstance(kappa, list):
            kappa = np.array(kappa, dtype=np.float64)
        if isinstance(Tmin, list):
            Tmin = np.array(Tmin, dtype=np.float64)
        if isinstance(Tmax, list):
            Tmax = np.array(Tmax, dtype=np.float64)
        # check shapes
        self._shape = self._check_shapes([DSa, DHa], [kappa, Tmin, Tmax])

        # check bounds
        check_bounds(DSa, 0, np.inf, 'DSa')
        check_bounds(DHa, 0, np.inf, 'DHa')
        check_bounds(kappa, 0, 1, 'kappa')
        check_bounds(Tmin, 0, np.inf, 'Tmin')
        check_bounds(Tmax, 0, np.inf, 'Tmax')
        check_bounds(Tmax-Tmin, eps, np.inf, 'Tmax-Tmin')

        self.DSa = DSa
        self.DHa = DHa
        self.kappa = kappa
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.name = name

    def eval(self, T):
        return self.kappa*kB*T/h*np.exp((self.DSa-self.DHa/T)/R)


class CoefficientCLD(Coefficient):
    """_Abstract_ chain-length dependent (CLD) coefficient."""
    pass


class TerminationCompositeModel(CoefficientCLD):
    r"""Composite model for the termination rate coefficient.

    The chain-length dependence is given by:

    $$ k_t(i,j)=\sqrt{k_t(i,i) k_t(j,j)} $$

    with:

    $$ k_t(i,i)=\begin{cases}
    k_t(1,1)i^{-\alpha_S}& \text{if } i \leq i_{crit} \\
    k_t(1,1)i_{crit}^{-(\alpha_S-\alpha_L)}i^{-\alpha_L} & \text{if } i>i_{crit}
    \end{cases}
    $$

    where $k_t(1,1)$ is the temperature-dependent termination rate coefficient
    between two monomeric radicals, $i_{crit}$ is the critical chain length,
    $\alpha_S$ is the short-chain exponent, and $\alpha_L$ is the long-chain
    exponent.

    Parameters
    ----------
    k11 : Arrhenius | Eyring
        Temperature-dependent termination rate coefficient between two
        monomeric radicals, $k_t(1,1)$.
    icrit : int
        Sritical chain length, $i_{crit}$.
    aS : float
        Short-chain exponent, $\alpha_S$.
    aL : float
        Long-chain exponent, $\alpha_L$.
    name : str
        Name.
    """

    def __init__(self,
                 k11: Union[Arrhenius, Eyring],
                 icrit: int,
                 aS: float,
                 aL: float,
                 name: str = ''
                 ) -> None:

        check_bounds(icrit, 1, 200, 'icrit')
        check_bounds(aS, 0, 1, 'alpha_short')
        check_bounds(aL, 0, 0.5, 'alpha_long')

        self.k11 = k11
        self.icrit = icrit
        self.aS = aS
        self.aL = aL
        self.name = name

    def eval(self,
             T: FloatOrArray,
             i: IntOrArrayLike,
             j: IntOrArrayLike
             ) -> FloatOrArray:
        """Evaluate coefficient at given SI conditions, without checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        i : IntOrArrayLike
            Chain length of 1st radical.
        j : IntOrArrayLike
            Chain length of 2nd radical.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """

        k11 = self.k11.eval(T)
        aS = self.aS
        aL = self.aL
        icrit = self.icrit

        def ktii(i):
            return np.where(i <= icrit,
                            k11*i**(-aS),
                            k11*icrit**(-aS+aL)*i**(-aL))

        return np.sqrt(ktii(i)*ktii(j))

    def __call__(self,
                 T: FloatOrArray,
                 i: IntOrArrayLike,
                 j: IntOrArrayLike,
                 Tunit: Literal['C', 'K'] = 'C'
                 ) -> FloatOrArray:
        r"""Evaluate coefficient at given conditions, including unit
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
        TK = self.k11._convert_check_temperature(T, Tunit)
        return self.eval(TK, i, j)


class DIPPR(CoefficientT):
    """[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)
    temperature-dependent coefficient."""

    def __init__(self,
                 A: FloatOrArrayLike,
                 B: FloatOrArrayLike,
                 C: FloatOrArrayLike,
                 D: FloatOrArrayLike,
                 E: FloatOrArrayLike,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = +np.inf,
                 name: str = ''
                 ) -> None:

        # convert lists to arrays
        if isinstance(A, list):
            A = np.array(A, dtype=np.float64)
        if isinstance(B, list):
            B = np.array(B, dtype=np.float64)
        if isinstance(C, list):
            C = np.array(C, dtype=np.float64)
        if isinstance(D, list):
            D = np.array(D, dtype=np.float64)
        if isinstance(E, list):
            E = np.array(E, dtype=np.float64)
        if isinstance(Tmin, list):
            Tmin = np.array(Tmin, dtype=np.float64)
        if isinstance(Tmax, list):
            Tmax = np.array(Tmax, dtype=np.float64)

        # check shapes
        self._shape = self._check_shapes([A, B, C, D, E], [Tmin, Tmax])

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.name = name


class DIPPR100(DIPPR):
    r"""DIPPR-100 equation.

    The temperature dependence is given by:

    $$ Y = A + B T + C T^2 + D T^3 + E T^4 $$

    Parameters
    ----------
    A : FloatOrArrayLike
        Parameter of equation.
    B : FloatOrArrayLike
        Parameter of equation.
    C : FloatOrArrayLike
        Parameter of equation.
    D : FloatOrArrayLike
        Parameter of equation.
    E : FloatOrArrayLike
        Parameter of equation.
    Tmin : FloatOrArrayLike
        Lower temperature bound.
        Unit = K.
    Tmax : FloatOrArrayLike
        Upper temperature bound.
        Unit = K.
    name : str
        Name.
    """

    def eval(self, T):
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        return A + B * T + C * T**2 + D * T**3 + E * T**4


class DIPPR101(DIPPR):
    r"""DIPPR-101 equation.

    The temperature dependence is given by:

    $$ Y = \exp{\left(A + B / T + C \ln(T) + D T^E\right)} $$

    Parameters
    ----------
    A : FloatOrArrayLike
        Parameter of equation.
    B : FloatOrArrayLike
        Parameter of equation.
    C : FloatOrArrayLike
        Parameter of equation.
    D : FloatOrArrayLike
        Parameter of equation.
    E : FloatOrArrayLike
        Parameter of equation.
    Tmin : FloatOrArrayLike
        Lower temperature bound.
        Unit = K.
    Tmax : FloatOrArrayLike
        Upper temperature bound.
        Unit = K.
    name : str
        Name.
    """

    def eval(self, T):
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        return np.exp(A + B / T + C * np.log(T) + D * T**E)


class DIPPR105(DIPPR):
    r"""DIPPR-105 equation.

    The temperature dependence is given by:

    $$ Y = \frac{A}{B^{ \left( 1 + (1 - T / C)^D \right) }} $$

    Parameters
    ----------
    A : FloatOrArrayLike
        Parameter of equation.
    B : FloatOrArrayLike
        Parameter of equation.
    C : FloatOrArrayLike
        Parameter of equation.
    D : FloatOrArrayLike
        Parameter of equation.
    Tmin : FloatOrArrayLike
        Lower temperature bound.
        Unit = K.
    Tmax : FloatOrArrayLike
        Upper temperature bound.
        Unit = K.
    name : str
        Name.
    """

    def __init__(self,
                 A: FloatOrArrayLike,
                 B: FloatOrArrayLike,
                 C: FloatOrArrayLike,
                 D: FloatOrArrayLike,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = +np.inf,
                 name: str = ''
                 ) -> None:

        if isinstance(A, (list, np.ndarray)):
            E = [0.0]*len(A)
        else:
            E = 0.0
        super().__init__(A, B, C, D, E, Tmin, Tmax, name)

    def eval(self, T):
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        return A / B**(1 + (1 - T / C)**D)

# %%
