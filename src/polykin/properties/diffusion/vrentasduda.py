# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Iterable, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from numpy import exp
from scipy.constants import R as Rgas

from polykin.utils.tools import (check_bounds, check_in_set, check_valid_range,
                                 convert_check_temperature)
from polykin.utils.types import FloatArray, FloatArrayLike

# %%

__all__ = ['VrentasDudaBinary']

# %% Vrentas-Duda


class VrentasDudaBinary():
    r"""Vrentas-Duda free volume model for the diffusivity of binary polymer
    solutions.

    The solvent self-diffusion coefficient is given by:

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

    **References**

    *   Vrentas, J.S. and Duda, J.L. (1977), J. Polym. Sci. Polym. Phys. Ed.,
        15: 403-416.

    Parameters
    ----------
    D0 : float
        Pre-exponential factor.
        Unit = L²/T.
    E : float
        Activation energy required to overcome the atractive forces
        between neighboring molecules.
        Units = J/mol/K.
    v1star : float
        Specific volume of solvent at 0 K.
        Unit = L³/M.
    v2star : float
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

    Examples
    --------
    Estimate the mutual and self-diffusion coefficient of toluene in
    polyvinylacetate at 20 wt% toluene and 25°C.

    >>> from polykin.properties.diffusion import VrentasDudaBinary
    >>> d = VrentasDudaBinary(
    ...     D0=4.82e-4, E=0., v1star=0.917, v2star=0.728, z=0.82,
    ...     K11=1.45e-3, K12=4.33e-4, K21=-86.32, K22=-258.2, X=0.5,
    ...     unit='cm²/s',
    ...     name='Tol(1)/PVAc(2)')

    >>> D = d(0.2, 25., Tunit='C')
    >>> print(f"D = {D:.2e} {d.unit}")
    D = 3.79e-08 cm²/s

    >>> D1 = d(0.2, 25., Tunit='C', selfd=True)
    >>> print(f"D1 = {D1:.2e} {d.unit}")
    D1 = 7.40e-08 cm²/s

    """  # noqa: E501

    D0: float
    E: float
    v1star: float
    v2star: float
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
                 v1star: float,
                 v2star: float,
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

        check_bounds(D0, 0., np.inf, 'D0')
        check_bounds(E, 0., np.inf, 'E')
        check_bounds(v1star, 0., np.inf, 'v1star')
        check_bounds(v2star, 0., np.inf, 'v2star')
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
        self.v1star = v1star
        self.v2star = v2star
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
            f"name: {self.name}\n"
            f"unit: {self.unit}\n"
            f"D0:   {self.D0}\n"
            f"E:    {self.E}\n"
            f"v1*:  {self.v1star}\n"
            f"v2*:  {self.v2star}\n"
            f"ξ:    {self.z}\n"
            f"K11:  {self.K11}\n"
            f"K12:  {self.K12}\n"
            f"K21:  {self.K21}\n"
            f"K22:  {self.K22}\n"
            f"Tg1:  {self.Tg1}\n"
            f"Tg2:  {self.Tg2}\n"
            f"𝛾:    {self.y}\n"
            f"𝜒:    {self.X}"
        )

    def __call__(self,
                 w1: Union[float, FloatArrayLike],
                 T: Union[float, FloatArrayLike],
                 Tunit: Literal['C', 'K'] = 'K',
                 selfd: bool = False
                 ) -> Union[float, FloatArray]:
        r"""Evaluate solvent self-diffusion, $D_1$, or mutual diffusion
        coefficient, $D$, at given solvent content and temperature, including
        unit conversion and range check.

        Parameters
        ----------
        w1 : float | FloatArrayLike
            Mass fraction of solvent.
            Unit = kg/kg.
        T : float | FloatArrayLike
            Temperature.
            Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.
        selfd: bool
            Switch result between mutual diffusion coefficient (if `False`) and
            self-diffusion coefficient (if `True`).

        Returns
        -------
        float | FloatArray
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
              w1: Union[float, FloatArray],
              T: Union[float, FloatArray]
              ) -> Union[float, FloatArray]:
        r"""Evaluate solvent self-diffusion coefficient, $D_1$, at given SI
        conditions, without unit conversions or checks.

        Parameters
        ----------
        w1 : float | FloatArray
            Mass fraction of solvent.
            Unit = kg/kg.
        T : float | FloatArray
            Temperature.
            Unit = K.

        Returns
        -------
        float | FloatArray
            Solvent self-diffusion coefficient, $D_1$.
        """

        D0 = self.D0
        E = self.E
        V1star = self.v1star
        V2star = self.v2star
        z = self.z
        K11 = self.K11
        K12 = self.K12
        K21 = self.K21
        K22 = self.K22
        Tg1 = self.Tg1
        Tg2 = self.Tg2
        y = self.y

        w2 = 1 - w1
        D1 = D0*exp(-E/(Rgas*T)) * \
            exp(-(w1*V1star + w2*z*V2star) /
                (w1*(K11/y)*(K21 - Tg1 + T) + w2*(K12/y)*(K22 - Tg2 + T)))
        return D1

    def mutual(self,
               w1: Union[float, FloatArray],
               T: Union[float, FloatArray]
               ) -> Union[float, FloatArray]:
        r"""Evaluate mutual diffusion coefficient, $D$, at given SI conditions,
        without unit conversions or checks.

        Parameters
        ----------
        w1 : float | FloatArray
            Mass fraction of solvent.
            Unit = kg/kg.
        T : float | FloatArray
            Temperature.
            Unit = K.

        Returns
        -------
        float | FloatArray
            Mutual diffusion coefficient, $D$.
        """
        D1 = self.selfd(w1, T)
        X = self.X
        D = D1 * (1 - w1)**2 * (1 - 2*X*w1)
        return D

    def plot(self,
             T: Union[float, FloatArrayLike],
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
        T : float | FloatArrayLike
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
        ax.set_xlabel("$w_1$ [kg/kg]")
        if selfd:
            Dsymbol = "$D_1$"
        else:
            Dsymbol = "$D$"
        ax.set_ylabel(Dsymbol + f" [{self.unit}]")
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        if return_objects:
            return (fig, ax)

    def fit(self):
        return NotImplemented


class VrentasDudaMulticomponent():
    pass
