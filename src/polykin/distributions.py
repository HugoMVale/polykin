# %%

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from utils import check_bounds, check_type, check_in_set


class Distribution(ABC):
    """Abstract class for all chain-length distributions."""

    def __init__(self, DPn: int = 100, M0: float = 100.0, name: str = ""):
        """Initialize chain-length distribution.
        Args:
            DPn (int, optional): Number-average degree of polymerization, $DP_n$.
            M0 (float, optional): Molar mass of the repeating unit, $M_0$.
            name (str, optional): Name.
        """
        self.DPn = DPn
        self.M0 = M0
        self.name = name

    def __call__(self, x: int | float | list | np.ndarray,
                 dist: str = "mass", unit_x: str = "chain_length"):
        """Evaluate chain-length distribution.

        Args:
            x (int | float | list | np.ndarray): Chain length or molar mass.
            dist (str, optional): Type of distribution. Options: 'number', 'mass', 'gpc'.
            unit_x (str, optional): Unit of variable `x`. Options: 'chain_length' or 'molar_mass'.

        Returns:
            (float | np.ndarray): chain-length probability.
        """

        # Select unit_x
        check_in_set(unit_x, {'chain_length', 'molar_mass'}, 'unit_x')
        x = self._list_to_array(x)
        if unit_x == "molar_mass":
            x /= self.DPn

        # Select distribution
        check_in_set(dist, {'number', 'mass', 'gpc'}, 'dist')
        if dist == "number":
            factor = 1
        elif dist == "mass":
            factor = x / self.DPn
        elif dist == "gpc":
            factor = x**2 / self._moment(2)
        else:
            raise ValueError

        return factor*self._pmf(x)

    @property
    def M0(self) -> float:
        """Molar mass of the repeating unit, $M_0$."""
        return self.__M0

    @M0.setter
    def M0(self, M0: float):
        """Molar mass of the repeating unit, $M_0$."""
        self.__M0 = check_bounds(M0, 0.0, np.Inf, "M0")

    @property
    def name(self) -> str:
        """Name of the distribution."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """Name of the distribution."""
        check_type(name, str, "name")
        if name != "":
            full_name = f"{name} ({self.__class__.__name__})"
        else:
            full_name = self.__class__.__name__
        self.__name = full_name

    @property
    def DPn(self) -> float:
        """Number-average degree of polymerization, $DP_n$."""
        return self.__DPn

    @DPn.setter
    def DPn(self, DPn: int = 100):
        """Set average degree of polymerization."""
        self.__DPn = check_bounds(DPn, 2, np.Inf, "DPn")

    @property
    def DPw(self) -> float:
        """Mass-average degree of polymerization, $DP_w$."""
        return self._moment(2) / self._moment(1)

    @property
    def DPz(self) -> float:
        """z-average degree of polymerization, $DP_z$."""
        return self._moment(3) / self._moment(2)

    @property
    def PDI(self) -> float:
        """Polydispersity index, $DP_w/DP_n$."""
        return self.DPw / self.DPn

    @property
    def Mn(self) -> float:
        """Number-average molar mass, $M_n$."""
        return self.M0 * self.DPn

    @property
    def Mw(self) -> float:
        """Weight-average molar mass, $M_w$."""
        return self.M0 * self.DPw

    @property
    def Mz(self) -> float:
        """z-average molar mass, $M_z$."""
        return self.M0 * self.DPz

    @abstractmethod
    def _pmf(self, length) -> float | np.ndarray:
        return 0.0

    @abstractmethod
    def _cdf(self, length) -> float | np.ndarray:
        return 0.0

    @abstractmethod
    def _moment(self, order: int) -> float:
        return 0.0

    @abstractmethod
    def _xrange_auto(self) -> tuple:
        return ()

    @staticmethod
    def _list_to_array(length) -> np.ndarray:
        if isinstance(length, list):
            length = np.asarray(length)
        return length

    def __str__(self) -> str:
        """Display key properties of the chain-length distribution."""
        return f"name: {self.name}\n" + \
            f"DPn:  {self.DPn:.1f}\n" + \
            f"DPw:  {self.DPw:.1f}\n" + \
            f"DPz:  {self.DPw:.1f}\n" + \
            f"PDI:  {self.PDI:.2f}\n" + \
            f"Mn:   {self.Mn:,.0f}\n" + \
            f"Mw:   {self.Mw:,.0f}\n" + \
            f"Mz:   {self.Mz:,.0f}"

    def plot(self, dist: str = 'mass', unit_x: str = 'chain_length',
             xscale: str = 'auto', xrange: tuple = (), ax=None):
        """Plot the chain-length distribution.

        Args:
            dist (str, optional): Type of distribution. Options: 'number', 'mass', 'gpc'.
            unit_x (str, optional): Unit of variable `x`. Options: 'chain_length' or 'molar_mass'.
            xscale (str, optional): x-axis scale. Options: 'linear', 'log', 'auto'.
            xrange (tuple, optional): x-axis range.
            ax (matplotlib.axes, optional): Matplotlib axes object.

        Returns:
            (matplotlib.axes):  Matplotlib axes object.
        """

        # Check inputs
        check_in_set(unit_x, {'chain_length', 'molar_mass'}, 'unit_x')
        check_in_set(xscale, {'linear', 'log', 'auto'}, 'xscale')
        check_type(dist, (str, list, tuple), 'dist')
        if isinstance(dist, str):
            dist = [dist]

        # Create axis if none is provided
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(f"Distribution: {self.name}")
            self.fig = fig

        # x-axis
        if len(xrange) != 2:
            xrange = self._xrange_auto()
        if xscale == 'log' or (xscale == 'auto' and set(dist) == {'gpc'}):
            x = np.geomspace(xrange[0], xrange[1], 100)
            xscale = 'log'
        else:
            x = np.linspace(xrange[0], xrange[1], 100)
            xscale = 'linear'
        if unit_x == "chain_length":
            xp = x
            label_x = "Chain length"
        elif unit_x == "molar_mass":
            xp = x * self.M0
            label_x = "Molar mass"
        else:
            raise ValueError

        # y-axis
        for item in dist:
            y = self(x, dist=item, unit_x="chain_length")
            ax.plot(xp, y, label=item)

        # Other properties
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        ax.grid(True)
        ax.set_xlabel(label_x)
        ax.set_ylabel("Relative abundance")
        ax.set_xscale(xscale)

        return ax


class Flory(Distribution):
    """Flory-Schulz (aka most-probable) chain-length distribution, where the
    *number* probability mass function is given by:

    $$ x(k) = (1-a)a^{k-1} $$

    with $a=1-1/DP_n$.
    """

    def _pmf(self, k):
        a = 1 - 1 / self.DPn
        return (1 - a) * a ** (k - 1)

    def _cdf(self, k):
        return 0.0

    def _xrange_auto(self):
        return (1, 10*self.DPn)

    def _moment(self, order: int):
        if order == 0:
            result = 1
        elif order == 1:
            result = self.DPn
        elif order == 2:
            result = self.DPn * (2 * self.DPn - 1)
        elif order == 3:
            result = self.DPn * (1 + 6 * self.DPn * (self.DPn - 1))
        else:
            raise ValueError("Not defined for order>3.")
        return result


class Poisson(Distribution):
    """Poisson chain-length distribution, where the *number* probability mass
    function is given by:

    $$ x(k) = {a^{k-1} e^{-a}} / {\Gamma(k)} $$

    with $a=DP_n-1$.
    """

    def _pmf(self, k):
        a = self.DPn - 1
        return np.exp((k - 1)*np.log(a) - a - sc.gammaln(k))

    def _cdf(self, s, dist):
        a = self.DPn - 1
        if dist == 0:
            result = sc.gammaincc(s, a)
        elif dist == 1:
            result = ((a + 1)*sc.gammaincc(s, a) -
                      np.exp(s*np.log(a) - a - sc.gammaln(s)))/self.DPn
        else:
            raise ValueError("Not defined for order>2.")
        return result

    def _xrange_auto(self):
        return (max(1, self.DPn/2-10), 1.5*self.DPn+10)

    def _moment(self, order: int):
        a = self.DPn - 1
        if order == 0:
            result = 1
        elif order == 1:
            result = self.DPn
        elif order == 2:
            result = a**2 + 3*a + 1
        elif order == 3:
            result = a**3 + 6*a**2 + 7*a + 1
        else:
            raise ValueError("Not defined for order>3.")
        return result


# %%
p = Poisson(10)
x = np.linspace(1, 20, 100)
cdf0 = p._cdf(x, 0)
cdf1 = p._cdf(x, 1)

plt.plot(x, cdf0, label='number')
plt.plot(x, cdf1, label='mass')
