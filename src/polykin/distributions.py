# %%

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Distribution(ABC):
    """Abstract class for all chain-length distributions."""

    def __init__(self, DPn: int = 100, molar_mass: float = 100.0, name: str = ""):
        self.DPn = DPn
        self.molar_mass = molar_mass
        if name != "":
            self.name = name
        else:
            self.name = self.__class__.__name__

    def __call__(self, x, dist: str = "mass", unit_x: str = "chain_length"):

        # Select unit_x
        if unit_x == "chain_length":
            pass
        elif unit_x == "molar_mass":
            x = self._list_to_array(x) / self.DPn
        else:
            raise ValueError

        # Select distribution
        if dist == "number":
            w = self._dist_number(x)
        elif dist == "mass":
            w = self._dist_mass(x)
        elif dist == "gpc":
            w = self._dist_gpc(x)
        else:
            raise ValueError

        return w

    def _dist_number(self, length):
        return self._pmf(self._list_to_array(length))

    def _dist_mass(self, length):
        return self._dist_number(length) * length / self._DPn

    def _dist_gpc(self, length):
        return self._dist_number(length) * length**2 / self._moment(2)

    @property
    def DPn(self):
        """Number-average degree of polymerization, $DP_n$."""
        return self._DPn

    @DPn.setter
    def DPn(self, DPn: int = 100):
        """Set average degree of polymerization."""
        self._DPn = DPn

    @property
    def DPw(self):
        """Mass-average degree of polymerization, $DP_w$."""
        return self._moment(2) / self._moment(1)

    @property
    def DPz(self):
        """z-average degree of polymerization, $DP_z$."""
        return self._moment(3) / self._moment(2)

    @property
    def PDI(self):
        """Polydispersity index, $DP_w/DP_n$."""
        return self.DPw / self.DPn

    @property
    def Mn(self):
        """Number-average molar mass, $M_n$."""
        return self.molar_mass * self.DPn

    @property
    def Mw(self):
        """Weight-average molar mass, $M_w$."""
        return self.molar_mass * self.DPw

    @property
    def Mz(self):
        """z-average molar mass, $M_z$."""
        return self.molar_mass * self.DPz

    @abstractmethod
    def _pmf(self, length):
        return 0.0

    @abstractmethod
    def _cdf(self, length):
        return 0.0

    @abstractmethod
    def _moment(self, order: int):
        return 0.0

    @staticmethod
    def _list_to_array(length):
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

    def plot(self, dist: str = "mass", unit_x: str = "chain_length", ax=None):
        """Plot the chain-length distribution."""

        # Create axis if none is provided
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(f"Distribution: {self.name}")
            self.fig = fig

        # Plot distribution
        x = np.linspace(1, 10 * self._DPn, 100)
        y = self(x, dist=dist, unit_x="chain_length")
        if unit_x == "molar_mass":
            x = x * self.molar_mass
            label_x = "Molar mass"
        elif unit_x == "chain_length":
            label_x = "Chain length"
        else:
            raise ValueError
        ax.plot(x, y, label=dist)
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        ax.grid(True)
        ax.set_xlabel(label_x)
        ax.set_ylabel("Relative abundance")

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
            raise ValueError
        return result


class Poisson(Distribution):
    def _pmf(self, i):
        a = 2.0 / (self.DPn + 1)
        w = a**2 * i * (1 - a) ** (i - 1)
        return w


# %%

d = Flory(142)
d.show
d.plot(dist="mass")
