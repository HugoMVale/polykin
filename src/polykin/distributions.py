# %%

from abc import ABC, abstractmethod
from utils import check_bounds, check_type, check_in_set
import numpy as np
import matplotlib.pyplot as plt


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
        if unit_x == "molar_mass":
            x = self._list_to_array(x) / self.DPn

        # Select distribution
        check_in_set(dist, {'number', 'mass', 'gpc'}, 'dist')
        if dist == "number":
            result = self._dist_number(x)
        elif dist == "mass":
            result = self._dist_mass(x)
        elif dist == "gpc":
            result = self._dist_gpc(x)
        else:
            raise ValueError

        return result

    def _dist_number(self, length):
        return self._pmf(self._list_to_array(length))

    def _dist_mass(self, length):
        return self._dist_number(length) * length / self.DPn

    def _dist_gpc(self, length):
        return self._dist_number(length) * length**2 / self._moment(2)

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
        self.__DPn = check_bounds(DPn, 1, np.Inf, "M0")

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
    def _pmf(self, length) -> float:
        return 0.0

    @abstractmethod
    def _cdf(self, length) -> float:
        return 0.0

    @abstractmethod
    def _moment(self, order: int) -> float:
        return 0.0

    @staticmethod
    def _list_to_array(length) -> np.ndarray:
        if isinstance(length, list):
            length = np.asarray(length)
        return length

    @property
    def show(self):
        """Show key properties of the chain-length distribution."""
        print(f"DPn: {self.DPn:.1f}")
        print(f"DPw: {self.DPw:.1f}")
        print(f"DPz: {self.DPw:.1f}")
        print(f"PDI: {self.PDI:.2f}")
        print(f"Mn:  {self.Mn:,.0f}")
        print(f"Mw:  {self.Mw:,.0f}")
        print(f"Mz:  {self.Mz:,.0f}")

    def plot(self, dist: str = "mass", unit_x: str = "chain_length",
             xscale: str = 'linear', ax=None):
        """Plot the chain-length distribution."""

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
        if xscale == 'log' or (xscale == 'auto' and set(dist) == {'gpc'}):
            x = np.geomspace(1, 10 * self.DPn, 100)
            xscale = 'log'
        else:
            x = np.linspace(1, 10 * self.DPn, 100)
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
    """Flory-Schulz (aka most-probable) chain-length distribution."""

    def _pmf(self, i):
        a = 1 - 1 / self.DPn
        result = (1 - a) * a ** (i - 1)
        return result

    def _cdf(self, i):
        return 0.0

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
    def _pmf(self, i):
        a = 2.0 / (self.DPn + 1)
        w = a**2 * i * (1 - a) ** (i - 1)
        return w


# %%

d = Flory(142)
d.show
d.plot(dist="mass", xscale='linear')
d.plot(dist="mass", xscale='log')
d.plot(dist="mass", xscale='auto')
d.plot(dist="gpc", xscale='auto')
