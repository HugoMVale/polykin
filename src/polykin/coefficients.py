"""
PolyKin `coefficients` provides classes to create and visualize the types of
kinetic coefficients and physical property correlations most often found in
polymer reactor models.

"""

from polykin.utils import check_bounds, check_type, check_in_set, \
    check_shapes, check_valid_range, convert_check_temperature, \
    FloatOrArray, FloatOrArrayLike, IntOrArrayLike, IntOrArray, \
    eps, ShapeError
from polykin.base import Base

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.constants import h, R, Boltzmann as kB
from abc import ABC, abstractmethod
from typing import Union, Literal


class Coefficient(Base, ABC):
    """_Abstract_ class for all coefficients, c(...)."""

    def __init__(self) -> None:
        self._shape = None

    @property
    def shape(self) -> Union[tuple[int, ...], None]:
        """Shape of underlying coefficient array."""
        return self._shape

    @abstractmethod
    def __call__(self, *args) -> FloatOrArray:
        pass

    @abstractmethod
    def eval(self, *args) -> FloatOrArray:
        pass


class CoefficientX1(Coefficient):
    r"""_Abstract_ class for 1-argument coefficient, $c(x)$.
    """
    pass


class CoefficientX2(Coefficient):
    r"""_Abstract_ class for 2-arguments coefficient, $c(x, y)$.
    """
    pass


class CoefficientT(CoefficientX1):
    """_Abstract_ temperature-dependent coefficient."""

    Tmin: FloatOrArray
    Tmax: FloatOrArray
    unit: str
    symbol: str

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
        TK = convert_check_temperature(T, Tunit, self.Tmin, self.Tmax)
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

    def plot(self,
             kind: Literal['linear', 'semilogy', 'Arrhenius'] = 'linear',
             Trange: Union[tuple[float, float], None] = None,
             Tunit: Literal['C', 'K'] = 'K',
             title: Union[str, None] = None,
             axes: Union[plt.Axes, None] = None
             ) -> None:
        """Plot the coefficient as a function of temperature.

        Parameters
        ----------
        kind : Literal['linear', 'semilogy', 'Arrhenius']
            Kind of plot to be generated.
        Trange : tuple[float, float] | None
            Temperature range for x-axis. If `None`, the validity range
            (Tmin, Tmax) will be used. If no validity range was defined, the
            range will fall back to 0-100°C.
        Tunit : Literal['C', 'K']
            Temperature unit.
        title : str | None
            Title of plot. If `None`, the object name will be used.
        axes : list[plt.Axes] | None
            Matplotlib Axes object.
        """

        # Check inputs
        check_in_set(kind, {'linear', 'semilogy', 'Arrhenius'}, 'kind')
        check_in_set(Tunit, {'K', 'C'}, 'Tunit')
        if Trange is not None:
            check_valid_range(Trange, 'Trange')

        # Plot objects
        if axes is None:
            ext_mode = False
            fig, ax = plt.subplots()
            self.fig = fig
            if title is None:
                title = self.name
            if title:
                fig.suptitle(title)
        else:
            ext_mode = True
            ax = axes

        # Plot labels
        Tsymbol = Tunit
        if Tunit == 'C':
            Tsymbol = '°' + Tsymbol
        xlabel = fr"$T$ [{Tsymbol}]"
        ylabel = fr"${self.symbol}$ [${self.unit}$]"
        if ext_mode:
            label = ylabel
            ylabel = "$c(T)$"
        else:
            label = None
        Tunit_range = Tunit
        if kind == 'Arrhenius':
            Tunit = 'K'
            xlabel = r"$1/T$ [" + Tsymbol + r"$^{-1}$]"
            ylabel = r"$\ln$" + ylabel
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        # x-axis vector
        if Trange is not None:
            if Tunit_range == 'C':
                Trange = tuple(np.asarray(Trange) + 273.15)
        else:
            Trange = (np.min(self.Tmin), np.max(self.Tmax))
            if Trange == (0.0, np.inf):
                Trange = (273.15, 373.15)

        if self._shape:
            print("Plot method not yet implemented for array-like coefficients.")
        else:
            TK = np.linspace(Trange[0], Trange[1], 100)
            y = self.__call__(TK, 'K')
            TC = TK - 273.15
            if Tunit == 'C':
                T = TC
            else:
                T = TK
            if kind == 'linear':
                ax.plot(T, y, label=label)
            elif kind == 'semilogy':
                ax.semilogy(T, y, label=label)
            elif kind == 'Arrhenius':
                ax.semilogy(1/TK, y, label=label)

        if ext_mode:
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        return None


class Arrhenius(CoefficientT):
    r"""[Arrhenius](https://en.wikipedia.org/wiki/Arrhenius_equation) kinetic
    rate coefficient.

    This coefficient implements the following temperature dependence:

    $$ k(T)=k_0\exp\left[-\frac{E_a}{R}\left(\frac{1}{T}-\frac{1}{T_0} \\
        \right)\right] $$

    where $T_0$ is a convenient reference temperature, $E_a$ is the activation
    energy, and $k_0=k(T_0)$. In the limit $T\rightarrow+\infty$, the usual
    form of the Arrhenius equation with $k_0=A$ is recovered.

    Parameters
    ----------
    k0 : FloatOrArrayLike
        Coefficient value at the reference temperature, $k_0=k(T_0)$.
        Unit = `unit`.
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
    unit : str
        Unit of coefficient.
    symbol : str
        Symbol of coefficient $k$.
    name : str
        Name.
    """

    def __init__(self,
                 k0: FloatOrArrayLike,
                 EaR: FloatOrArrayLike,
                 T0: FloatOrArrayLike = np.inf,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = np.inf,
                 unit: str = '-',
                 symbol: str = 'k',
                 name: str = ''
                 ) -> None:

        # Convert lists to arrays
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

        # Check shapes
        self._shape = check_shapes([k0, EaR], [T0, Tmin, Tmax])

        # Check bounds
        check_bounds(k0, 0, np.inf, 'k0')
        check_bounds(EaR, -np.inf, np.inf, 'EaR')
        check_bounds(T0, 0, np.inf, 'T0')
        check_bounds(Tmin, 0, np.inf, 'Tmin')
        check_bounds(Tmax, 0, np.inf, 'Tmax')
        check_bounds(Tmax-Tmin, eps, np.inf, 'Tmax-Tmin')

        self.k0 = k0
        self.EaR = EaR
        self.T0 = T0
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.unit = check_type(unit, str, 'unit')
        self.symbol = check_type(symbol, str, 'symbol')
        self.name = name

    def __repr__(self) -> str:
        return \
            f"name:      {self.name}\n" \
            f"symbol:    {self.symbol}\n" \
            f"unit:      {self.unit}\n" \
            f"k0:        {self.k0}\n" \
            f"Ea/R [K]:  {self.EaR}\n" \
            f"T0   [K]:  {self.T0}\n" \
            f"Tmin [K]:  {self.Tmin}\n" \
            f"Tmax [K]:  {self.Tmax}"

    def __mul__(self, other):
        """Multipy Arrhenius coefficient(s).

        Create a new Arrhenius coefficient from a product of two Arrhenius
        coefficients with identical shapes or a product of an Arrhenius
        coefficient and a numerical constant.

        Parameters
        ----------
        other : Arrhenius | float | int
            Another Arrhenius coefficient or number.

        Returns
        -------
        Arrhenius
            Product coefficient.
        """
        if isinstance(other, Arrhenius):
            if self._shape == other._shape:
                return Arrhenius(k0=self.A*other.A,
                                 EaR=self.EaR + other.EaR,
                                 Tmin=np.maximum(self.Tmin, other.Tmin),
                                 Tmax=np.minimum(self.Tmax, other.Tmax),
                                 unit=f"{self.unit}·{other.unit}",
                                 symbol=f"{self.symbol}·{other.symbol}",
                                 name=f"{self.name}·{other.name}")
            else:
                raise ShapeError(
                    "Product of array-like coefficients requires identical shapes.")
        elif isinstance(other, (int, float)):
            return Arrhenius(k0=self.k0*other,
                             EaR=self.EaR,
                             T0=self.T0,
                             Tmin=self.Tmin,
                             Tmax=self.Tmax,
                             unit=self.unit,
                             symbol=f"{str(other)}·{self.symbol}",
                             name=f"{str(other)}·{self.name}")
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide Arrhenius coefficient(s).

        Create a new Arrhenius coefficient from a division of two Arrhenius
        coefficients with identical shapes or a division involving an Arrhenius
        coefficient and a numerical constant.

        Parameters
        ----------
        other : Arrhenius | float | int
            Another Arrhenius coefficient or number.

        Returns
        -------
        Arrhenius
            Quotient coefficient.
        """
        if isinstance(other, Arrhenius):
            if self._shape == other._shape:
                return Arrhenius(k0=self.A/other.A,
                                 EaR=self.EaR - other.EaR,
                                 Tmin=np.maximum(self.Tmin, other.Tmin),
                                 Tmax=np.minimum(self.Tmax, other.Tmax),
                                 unit=f"{self.unit}/{other.unit}",
                                 symbol=f"{self.symbol}/{other.symbol}",
                                 name=f"{self.name}/{other.name}")
            else:
                raise ShapeError(
                    "Division of array-like coefficients requires identical shapes.")
        elif isinstance(other, (int, float)):
            return Arrhenius(k0=self.k0/other,
                             EaR=self.EaR,
                             T0=self.T0,
                             Tmin=self.Tmin,
                             Tmax=self.Tmax,
                             unit=self.unit,
                             symbol=f"{self.symbol}/{str(other)}",
                             name=f"{self.name}/{str(other)}")
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Arrhenius(k0=other/self.k0,
                             EaR=-self.EaR,
                             T0=self.T0,
                             Tmin=self.Tmin,
                             Tmax=self.Tmax,
                             unit=f"1/{self.unit}",
                             symbol=f"{str(other)}/{self.symbol}",
                             name=f"{str(other)}/{self.name}")
        else:
            return NotImplemented

    @property
    def A(self) -> FloatOrArray:
        r"""Pre-exponential factor, $A=k_0 e^{E_a/(R T_0)}$."""
        return self.eval(np.inf)

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        return self.k0 * np.exp(-self.EaR*(1/T - 1/self.T0))


class Eyring(CoefficientT):
    r"""[Eyring](https://en.wikipedia.org/wiki/Eyring_equation) kinetic rate
    coefficient.

    This coefficient implements the following temperature dependence:

    $$ k(T)=\dfrac{\kappa k_B T}{h} \\
        \exp\left(\frac{\Delta S^\ddagger}{R}\right) \\
        \exp\left(-\frac{\Delta H^\ddagger}{R T}\right)$$

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
        Unit = dimensionless.
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
    """

    unit = 's^{-1}'

    def __init__(self,
                 DSa: FloatOrArrayLike,
                 DHa: FloatOrArrayLike,
                 kappa: FloatOrArrayLike = 1.0,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = np.inf,
                 symbol: str = 'k',
                 name: str = ''
                 ) -> None:

        # Convert lists to arrays
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

        # Check shapes
        self._shape = check_shapes([DSa, DHa], [kappa, Tmin, Tmax])

        # Check bounds
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
        self.symbol = check_type(symbol, str, 'symbol')
        self.name = name

    def __repr__(self) -> str:
        return \
            f"name:             {self.name}\n" \
            f"symbol:           {self.symbol}\n" \
            f"unit:             {self.unit}\n" \
            f"DSa [J/(mol·K)]:  {self.DSa}\n" \
            f"DHa [J/mol]:      {self.DHa}\n" \
            f"kappa [—]:        {self.kappa}\n" \
            f"Tmin [K]:         {self.Tmin}\n" \
            f"Tmax [K]:         {self.Tmax}"

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        return self.kappa * kB*T/h * np.exp((self.DSa - self.DHa/T)/R)


class CoefficientCLD(Coefficient):
    """_Abstract_ chain-length dependent (CLD) coefficient."""
    pass


class TerminationCompositeModel(CoefficientCLD):
    r"""Composite model for the termination rate coefficient between two
    radicals.

    This coefficient implements the chain-length dependence proposed by
    [Smith and Russel (2003)](https://doi.org/10.1002/mats.200390029):

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
    """

    Ysymbol = "k_t (i,j)"

    def __init__(self,
                 k11: Union[Arrhenius, Eyring],
                 icrit: int,
                 aS: float = 0.5,
                 aL: float = 0.2,
                 name: str = ''
                 ) -> None:
        r"""
        Parameters
        ----------
        k11 : Arrhenius | Eyring
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

        check_bounds(icrit, 1, 200, 'icrit')
        check_bounds(aS, 0, 1, 'alpha_short')
        check_bounds(aL, 0, 0.5, 'alpha_long')

        self.k11 = k11
        self.icrit = icrit
        self.aS = aS
        self.aL = aL
        self.name = name

    def __repr__(self) -> str:
        return \
            f"name:      {self.name}\n" \
            f"icrit:     {self.icrit}\n" \
            f"aS:        {self.aS}\n" \
            f"aL:        {self.aL}\n" \
            "- k11 -\n" + \
            self.k11.__repr__()

    def eval(self,
             T: FloatOrArray,
             i: IntOrArray,
             j: IntOrArray
             ) -> FloatOrArray:
        """Evaluate coefficient at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        i : IntOrArray
            Chain length of 1st radical.
        j : IntOrArray
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
                 T: FloatOrArrayLike,
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
        TK = convert_check_temperature(T, Tunit, self.k11.Tmin, self.k11.Tmax)
        if isinstance(i, list):
            i = np.array(i, dtype=np.int32)
        if isinstance(j, list):
            j = np.array(j, dtype=np.int32)
        return self.eval(TK, i, j)


class PropagationHalfLength(CoefficientCLD):
    r"""Half-length model for the decay of the propagation rate coefficient
    with chain length.

    This coefficient implements the chain-length dependence proposed by
    [Smith et al. (2005)](https://doi.org/10.1016/j.eurpolymj.2004.09.002):

    $$ k_p(i) = k_p \left[ 1+ (C - 1)\exp{\left (-\frac{\ln 2}{i_{1/2}} (i-1) \
    \right )} \right] $$

    where $k_p=k_p(\infty)$ is the long-chain value of the propagation rate
    coefficient, $C\ge 1$ is the ratio $k_p(1)/k_p$ and $(i_{1/2}+1)$ is the
    hypothetical chain-length at which the difference $k_p(1) - k_p$ is halved.
    """

    def __init__(self,
                 kp: Union[Arrhenius, Eyring],
                 C: float = 10.,
                 ihalf: float = 1.0,
                 name: str = ''
                 ) -> None:
        r"""
        Parameters
        ----------
        kp : Arrhenius | Eyring
            Long-chain value of the propagation rate coefficient, $k_p$.
        C : float
            Ratio of the propagation coefficients of a monomeric radical and a 
            long-chain radical, $C$.
        ihalf : float
            Half-length, $i_{i/2}$.
        name : str
            Name.
        """

        check_bounds(C, 1., 100., 'C')
        check_bounds(ihalf, 0.1, 10, 'ihalf')

        self.kp = kp
        self.C = C
        self.ihalf = ihalf
        self.name = name

    def __repr__(self) -> str:
        return \
            f"name:      {self.name}\n" \
            f"C:         {self.C}\n" \
            f"ihalf:     {self.ihalf}\n" \
            "- kp -\n" + \
            self.kp.__repr__()

    def eval(self,
             T: FloatOrArray,
             i: IntOrArray,
             ) -> FloatOrArray:
        """Evaluate coefficient at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        i : IntOrArray
            Chain length of radical.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """
        kp = self.kp.eval(T)
        C = self.C
        ihalf = self.ihalf

        return kp*(1 + (C - 1)*np.exp(-np.log(2)*(i - 1)/ihalf))

    def __call__(self,
                 T: FloatOrArrayLike,
                 i: IntOrArrayLike,
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
            Chain length of radical.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """
        TK = convert_check_temperature(T, Tunit, self.kp.Tmin, self.kp.Tmax)
        if isinstance(i, list):
            i = np.array(i, dtype=np.int32)
        return self.eval(TK, i)


class DIPPR(CoefficientT):
    """_Abstract_ class for all [DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)
    temperature-dependent equations."""
    pass


class DIPPRP5(DIPPR):
    """_Abstract_ class for DIPPR equations with 5 parameters (A-E)."""

    def __init__(self,
                 A: FloatOrArrayLike,
                 B: FloatOrArrayLike,
                 C: FloatOrArrayLike,
                 D: FloatOrArrayLike,
                 E: FloatOrArrayLike,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = np.inf,
                 unit: str = '-',
                 symbol: str = 'Y',
                 name: str = ''
                 ) -> None:
        r"""
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
        unit : str
            Unit of output variable $Y$.
        symbol : str
            Symbol of output variable $Y$.
        name : str
            Name.
        """

        # Convert lists to arrays
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

        # Check shapes
        self._shape = check_shapes([A, B, C, D, E], [Tmin, Tmax])

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.unit = check_type(unit, str, 'unit')
        self.symbol = check_type(symbol, str, 'symbol')
        self.name = name

    def __repr__(self) -> str:
        return \
            f"name:      {self.name}\n" \
            f"symbol:    {self.symbol}\n" \
            f"unit:      {self.unit}\n" \
            f"A:         {self.A}\n" \
            f"B:         {self.B}\n" \
            f"C:         {self.C}\n" \
            f"D:         {self.D}\n" \
            f"E:         {self.E}\n" \
            f"Tmin [K]:  {self.Tmin}\n" \
            f"Tmax [K]:  {self.Tmax}"


class DIPPRP4(DIPPRP5):
    """_Abstract_ class for DIPPR equations with 4 parameters (A-D)."""

    def __init__(self,
                 A: FloatOrArrayLike,
                 B: FloatOrArrayLike,
                 C: FloatOrArrayLike,
                 D: FloatOrArrayLike,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = np.inf,
                 unit: str = '-',
                 symbol: str = 'Y',
                 name: str = ''
                 ) -> None:
        r"""
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
        unit : str
            Unit of output variable $Y$.
        symbol : str
            Symbol of output variable $Y$.
        name : str
            Name.
        """

        if isinstance(A, (list, np.ndarray)):
            E = [0.0]*len(A)
        else:
            E = 0.0
        super().__init__(A, B, C, D, E, Tmin, Tmax, unit, symbol, name)

    def __repr__(self) -> str:
        return \
            f"name:      {self.name}\n" \
            f"symbol:    {self.symbol}\n" \
            f"unit:      {self.unit}\n" \
            f"A:         {self.A}\n" \
            f"B:         {self.B}\n" \
            f"C:         {self.C}\n" \
            f"D:         {self.D}\n" \
            f"Tmin [K]:  {self.Tmin}\n" \
            f"Tmax [K]:  {self.Tmax}"


class DIPPR100(DIPPRP5):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-100 equation.

    This equation implements the following temperature dependence:

    $$ Y = A + B T + C T^2 + D T^3 + E T^4 $$

    """

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        return A + B * T + C * T**2 + D * T**3 + E * T**4


class DIPPR101(DIPPRP5):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-101 equation.

    This equation implements the following temperature dependence:

    $$ Y = \exp{\left(A + B / T + C \ln(T) + D T^E\right)} $$

    """

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        return np.exp(A + B / T + C * np.log(T) + D * T**E)


class DIPPR105(DIPPRP4):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-105 equation.

    This equation implements the following temperature dependence:

    $$ Y = \frac{A}{B^{ \left( 1 + (1 - T / C)^D \right) }} $$

    """

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        return A / B**(1 + (1 - T / C)**D)

# %% Functions


def plotcoeffs(coeffs: list[CoefficientT],
               kind: Literal['linear', 'semilogy', 'Arrhenius'] = 'linear',
               title: Union[str, None] = None,
               **kwargs
               ) -> Figure:
    """Plot a list of temperature-dependent coefficients in a combined plot.

    Parameters
    ----------
    coeffs : list[CoefficientT]
        List of coefficients to be ploted together.
    kind : Literal['linear', 'semilogy', 'Arrhenius']
        Kind of plot to be generated.
    title : str | None
        Title of plot.

    Returns
    -------
    Figure
        Matplotlib Figure object holding the combined plot.
    """

    # Create matplotlib objects
    fig, ax = plt.subplots()

    # Title
    if title is None:
        title = "Coefficient overlay"
    fig.suptitle(title)

    # Draw plots sequentially
    for c in coeffs:
        c.plot(kind=kind, axes=ax, **kwargs)

    return fig
