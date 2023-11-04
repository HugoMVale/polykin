# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.utils import check_bounds, check_in_set, check_valid_range, \
    convert_check_temperature, \
    FloatOrArray, FloatOrArrayLike

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from scipy.constants import R as Rgas
from typing import Literal, Optional
from collections.abc import Iterable

# %%

__all__ = ['VrentasDudaBinary']

# %% Vrentas-Duda


class VrentasDudaBinary():
    r"""Vrentas-Duda free volume model for the diffusivity of binary polymer
    solutions.

    The solvent self-diffusion coefficient is given by
    [Vrentas & Duda (1977)](https://doi.org/10.1002/pol.1977.180150302):

    $$ D_1 = D_0 e^{\left(-\frac{E}{RT}\right)}
       \exp\left[-\frac{\gamma (w_1\hat{V}_1^* + w_2 \xi \hat{V}_2^*)}
       {w_1 K_{11}(K_{21}-T_{g1}+T) + w_2 K_{12}(K_{22}-T_{g2}+T)}\right] $$

    and the mutual diffusion coefficient is given by:

    $$ D = D_1 (1 - w_1)^2 (1 - 2\chi w_1) $$

    where $D_0$ is the pre-exponential factor,
    $E$ is the activation energy required to overcome the atractive forces
    between neighboring molecules,
    $K_{ij}$ are free-volume parameters,
    $T$ is the temperature,
    $T_{gi}$ is the glass-transition temperature of component $i$,
    $\hat{V}_i^*$ is the specific volume of component $i$ at 0 kelvin,
    $w_i$ is the mass fraction of compoenent $i$,
    $\gamma$ is the overlap factor,
    $\xi$ is the ratio between the critical volume of the polymer and the
    solvent jumping units,
    and $\chi$ is the Flory-Huggins' interaction parameter.

    Parameters
    ----------
    D0 : float
        Pre-exponential factor.
        Unit = L²/T.
    E : float
        Activation energy required to overcome the atractive forces
        between neighboring molecules.
        Units = J/mol/K.
    V1star : float
        Specific volume of solvent at 0 K.
        Unit = L³/M.
    V2star : float
        Specific volume of polymer at 0 K.
        Unit = L³/M.
    z : float
        Ratio between the critical volume of the polymer and the
        solvent jumping units, $\xi$.
    K11 : float
        Free-volume parameter of solvent.
        Unit = L³/M/K.
    K12 : float
        Free-volume parameter of polymer.
        Unit = L³/M/K.
    K21 : float
        Free-volume parameter of solvent.
        Unit = K.
    K22 : float
        Free-volume parameter of polymer.
        Unit = K.
    Tg1 : float
        Glas-transition temperature of solvent.
        Unit = K.
    Tg2 : float
        Glas-transition temperature of polymer.
        Unit = K.
    y : float
        Overlap factor, $\gamma$.
    X : float
        Flory-Huggings interaction parameter, $\chi$.
    unit : str
        Unit of diffusivity, by definition equal to L²/T.
    name : str
        Name.
    """  # noqa: E501

    D0: float
    E: float
    V1star: float
    V2star: float
    z: float
    K11: float
    K12: float
    K21: float
    K22: float
    Tg1: float
    Tg2: float
    y: float
    X: float
    unit: str

    def __init__(self,
                 D0: float,
                 E: float,
                 V1star: float,
                 V2star: float,
                 z: float,
                 K11: float,
                 K12: float,
                 K21: float,
                 K22: float,
                 Tg1: float = 0.,
                 Tg2: float = 0.,
                 y: float = 1.,
                 X: float = 0.5,
                 unit: str = 'm²/s',
                 name: str = ''
                 ) -> None:
        r"""Construct `VrentasDuda` with the given parameters."""

        check_bounds(D0, 0., np.inf, 'D0')
        check_bounds(E, 0., np.inf, 'E')
        check_bounds(V1star, 0., np.inf, 'V1star')
        check_bounds(V2star, 0., np.inf, 'V2star')
        check_bounds(z, 0., np.inf, 'z')
        check_bounds(K11, 0., np.inf, 'K11')
        check_bounds(K12, 0., np.inf, 'K12')
        check_bounds(K21, -np.inf, np.inf, 'K21')
        check_bounds(K22, -np.inf, np.inf, 'K22')
        check_bounds(Tg1, 0., np.inf, 'Tg1')
        check_bounds(Tg2, 0., np.inf, 'Tg2')
        check_bounds(y, 0.5, 1., 'y')
        check_bounds(X, -10, 10, 'X')

        self.D0 = D0
        self.E = E
        self.V1star = V1star
        self.V2star = V2star
        self.z = z
        self.K11 = K11
        self.K12 = K12
        self.K21 = K21
        self.K22 = K22
        self.Tg1 = Tg1
        self.Tg2 = Tg2
        self.y = y
        self.X = X
        self.unit = unit
        self.name = name

    def __repr__(self) -> str:
        return (
            f"name:     {self.name}\n"
            f"unit:     {self.unit}\n"
            f"D0:       {self.D0}\n"
            f"E:        {self.E}\n"
            f"V1*:      {self.V1star}\n"
            f"V2*:      {self.V2star}\n"
            f"ξ:        {self.z}\n"
            f"K11:      {self.K11}\n"
            f"K12:      {self.K12}\n"
            f"K21:      {self.K21}\n"
            f"K22:      {self.K22}\n"
            f"Tg1:      {self.Tg1}\n"
            f"Tg2:      {self.Tg2}\n"
            f"𝛾:        {self.y}\n"
            f"𝜒:        {self.X}"
        )

    def __call__(self,
                 w1: FloatOrArrayLike,
                 T: FloatOrArrayLike,
                 Tunit: Literal['C', 'K'] = 'K',
                 selfd: bool = False
                 ) -> FloatOrArray:
        r"""Evaluate solvent self-diffusion, $D_1$, or mutual diffusion
        coefficient, $D$, at given solvent content and temperature, including
        unit conversion and range check.

        Parameters
        ----------
        w1 : FloatOrArrayLike
            Mass fraction of solvent.
            Unit = kg/kg.
        T : FloatOrArrayLike
            Temperature.
            Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.
        selfd: bool
            Switch result between mutual diffusion coefficient (if `False`) and
            self-diffusion coefficient (if `True`).

        Returns
        -------
        FloatOrArray
            Solvent self-diffusion or mutual diffusion coefficient.
        """
        if isinstance(w1, (list, tuple)):
            w1 = np.array(w1, dtype=np.float64)

        check_bounds(w1, 0., 1., 'w1')

        TK = convert_check_temperature(T, Tunit)
        if selfd:
            return self.selfd(w1, TK)
        else:
            return self.mutual(w1, TK)

    def selfd(self,
              w1: FloatOrArray,
              T: FloatOrArray
              ) -> FloatOrArray:
        r"""Evaluate solvent self-diffusion coefficient, $D_1$, at given SI
        conditions, without unit conversions or checks.

        Parameters
        ----------
        w1 : FloatOrArray
            Mass fraction of solvent.
            Unit = kg/kg.
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Solvent self-diffusion coefficient, $D_1$.
        """

        D0 = self.D0
        E = self.E
        V1star = self.V1star
        V2star = self.V2star
        z = self.z
        K11 = self.K11
        K12 = self.K12
        K21 = self.K21
        K22 = self.K22
        Tg1 = self.Tg1
        Tg2 = self.Tg2
        y = self.y

        w2 = 1 - w1
        D1 = D0*np.exp(-E/(Rgas*T)) * \
            np.exp(-(w1*V1star + w2*z*V2star) /
                   (w1*(K11/y)*(K21 - Tg1 + T) + w2*(K12/y)*(K22 - Tg2 + T)))
        return D1

    def mutual(self,
               w1: FloatOrArray,
               T: FloatOrArray
               ) -> FloatOrArray:
        r"""Evaluate mutual diffusion coefficient, $D$, at given SI conditions,
        without unit conversions or checks.

        Parameters
        ----------
        w1 : FloatOrArray
            Mass fraction of solvent.
            Unit = kg/kg.
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Mutual diffusion coefficient, $D$.
        """
        D1 = self.selfd(w1, T)
        X = self.X
        D = D1 * (1 - w1)**2 * (1 - 2*X*w1)
        return D

    def plot(self,
             T: FloatOrArrayLike,
             w1range: tuple[float, float] = (0., 0.5),
             Tunit: Literal['C', 'K'] = 'K',
             selfd: bool = False,
             title: Optional[str] = None,
             ylim: Optional[tuple[float, float]] = None,
             axes: Optional[Axes] = None,
             return_objects: bool = False
             ) -> Optional[tuple[Optional[Figure], Axes]]:
        """Plot the mutual or self-diffusion coefficient as a function of
        solvent content and temperature.

        Parameters
        ----------
        T : FloatOrArrayLike
            Temperature.
            Unit = `Tunit`.
        w1range : tuple[float, float]
            Range of solvent mass fraction to be ploted.
            Unit = kg/kg.
        Tunit : Literal['C', 'K']
            Temperature unit.
        selfd: bool
            Switch result between mutual diffusion coefficient (if `False`) and
            self-diffusion coefficient (if `True`).
        title : str | None
            Title of plot. If `None`, the object name will be used.
        ylim : tuple[float, float] | None
            User-defined limit of y-axis. If `None`, the default settings of
            matplotlib are used.
        axes : Axes | None
            Matplotlib Axes object.
        return_objects : bool
            If `True`, the Figure and Axes objects are returned (for saving or
            further manipulations).

        Returns
        -------
        tuple[Figure | None, Axes] | None
            Figure and Axes objects if return_objects is `True`.
        """

        # Check inputs
        check_in_set(Tunit, {'K', 'C'}, 'Tunit')
        check_valid_range(w1range, 0., 1., 'w1range')

        # Plot objects
        if axes is None:
            fig, ax = plt.subplots()
            if title is None:
                title = self.name
            if title:
                fig.suptitle(title)
        else:
            fig = None
            ax = axes

        Tsymbol = Tunit
        if Tunit == 'C':
            Tsymbol = '°C'
        if not isinstance(T, Iterable):
            T = [T]

        w1 = np.linspace(*w1range, 100)
        for Ti in T:
            y = self.__call__(w1, Ti, Tunit, selfd)
            ax.semilogy(w1, y, label=f"{Ti}{Tsymbol}")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel(r"$w_1$" + " [kg/kg]")
        if selfd:
            Dsymbol = r"$D_1$"
        else:
            Dsymbol = r"$D$"
        ax.set_ylabel(Dsymbol + f" [{self.unit}]")
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        if return_objects:
            return (fig, ax)

    def fit(self):
        return NotImplemented


class VrentasDudaMulticomponent():
    pass
