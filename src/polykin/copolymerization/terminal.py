# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from polykin.kinetics import Arrhenius
from polykin.utils.math import convert_FloatOrVectorLike_to_FloatVector, eps
from polykin.utils.tools import check_bounds, check_in_set, custom_repr
from polykin.utils.types import (FloatArray, FloatArrayLike, FloatVectorLike,
                                 IntArrayLike)

from .binary import inst_copolymer_binary, kp_average_binary
from .copodataset import CopoDataset, DriftDataset, MayoDataset, kpDataset
from .multicomponent import convert_Qe_to_r

__all__ = ['TerminalModel']

# %% Models


class CopoModel(ABC):

    k1: Optional[Arrhenius]
    k2: Optional[Arrhenius]
    M1: str
    M2: str
    name: str
    data: list[CopoDataset]

    _pnames: tuple[str, ...]

    def __init__(self,
                 k1: Optional[Arrhenius],
                 k2: Optional[Arrhenius],
                 M1: str,
                 M2: str,
                 name: str
                 ) -> None:

        if M1 and M2 and M1.lower() != M2.lower():
            self.M1 = M1
            self.M2 = M2
        else:
            raise ValueError("`M1` and `M2` must be non-empty and different.")

        self.k1 = k1
        self.k2 = k2
        self.name = name
        self.data = []

    def __repr__(self) -> str:
        return custom_repr(self, ('name', 'M1', 'M2', 'k1', 'k2')
                           + self._pnames)

    @abstractmethod
    def ri(self,
           f1: Union[float, FloatArray]
           ) -> tuple[Union[float, FloatArray],
                      Union[float, FloatArray]]:
        """Return the evaluated reactivity ratios at the given conditions."""
        pass

    @abstractmethod
    def kii(self,
            f1: Union[float, FloatArray],
            T: float,
            Tunit,
            ) -> tuple[Union[float, FloatArray],
                       Union[float, FloatArray]]:
        """Return the evaluated homopropagation rate coefficients at the given
        conditions."""
        pass

    def F1(self,
           f1: Union[float, FloatArrayLike]
           ) -> Union[float, FloatArray]:
        r"""Calculate the instantaneous copolymer composition, $F_1$.

        The calculation is handled by
        [`inst_copolymer_binary`](inst_copolymer_binary.md).

        Parameters
        ----------
        f1 : float | FloatArrayLike
            Molar fraction of M1.

        Returns
        -------
        float | FloatArray
            Instantaneous copolymer composition, $F_1$.
        """

        if isinstance(f1, (list, tuple)):
            f1 = np.array(f1, dtype=np.float64)
        check_bounds(f1, 0., 1., 'f1')
        return inst_copolymer_binary(f1, *self.ri(f1))

    def kp(self,
           f1: Union[float, FloatArrayLike],
           T: float,
           Tunit: Literal['C', 'K'] = 'K'
           ) -> Union[float, FloatArray]:
        r"""Calculate the average propagation rate coefficient, $\bar{k}_p$.

        The calculation is handled by
        [`kp_average_binary`](kp_average_binary.md).

        Note
        ----
        This feature requires the attributes `k11` and `k22` to be defined.

        Parameters
        ----------
        f1 : float | FloatArrayLike
            Molar fraction of M1.
        T : float
            Temperature. Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        float | FloatArray
            Average propagation rate coefficient. Unit = L/(mol·s)
        """

        if isinstance(f1, (list, tuple)):
            f1 = np.array(f1, dtype=np.float64)
        check_bounds(f1, 0., 1., 'f1')
        return kp_average_binary(f1, *self.ri(f1), *self.kii(f1, T, Tunit))

    @property
    def azeotrope(self) -> Optional[float]:
        r"""Calculate the azeotrope composition.

        Returns
        -------
        float | None
            If an azeotrope exists, it returns its composition in terms of
            $f_1$.
        """

        # Check if system is trivial
        ri_check = np.array(self.ri(0.1) + self.ri(0.9))
        if (np.all(np.isclose(ri_check, 1., atol=1e-2))):
            print("Warning: Trivial system with r1~r2~1.")
            return None

        def fzero(f1):
            return inst_copolymer_binary(f1, *self.ri(f1)) - f1

        try:
            solution = root_scalar(f=fzero,
                                   bracket=(1e-4, 0.9999),
                                   method='brentq')
            if solution.converged:
                result = solution.root
            else:
                print(solution.flag)
                result = None
        except ValueError:
            result = None

        return result

    def drift(self,
              f10: Union[float, FloatVectorLike],
              x: Union[float, FloatVectorLike]
              ) -> FloatArray:
        r"""Calculate drift of comonomer composition in a closed system for a
        given total monomer conversion.

        In a closed binary system, the drift in monomer composition is given by
        the solution of the following differential equation:

        $$ \frac{\textup{d} f_1}{\textup{d}x} = \frac{f_1 - F_1}{1 - x} $$

        with initial condition $f_1(0)=f_{1,0}$, where $f_1$ and $F_1$ are,
        respectively, the instantaneous comonomer and copolymer composition of
        M1, and $x$ is the total molar monomer conversion.

        Parameters
        ----------
        f10 : float | FloatVectorLike
            Initial molar fraction of M1, $f_{1,0}=f_1(0)$.
        x : float | FloatVectorLike
            Value(s) of total monomer conversion values where the drift is to
            be evaluated.

        Returns
        -------
        FloatArray
            Monomer fraction of M1 at a given conversion, $f_1(x)$.
        """
        f10, x = convert_FloatOrVectorLike_to_FloatVector([f10, x], False)

        def df1dx(x_, f1_):
            F1_ = inst_copolymer_binary(f1_, *self.ri(f1_))
            return (f1_ - F1_)/(1. - x_ + eps)

        sol = solve_ivp(df1dx,
                        (0., max(x)),
                        f10,
                        t_eval=x,
                        method='LSODA',
                        vectorized=True,
                        atol=1e-4,
                        rtol=1e-4)
        if sol.success:
            result = sol.y
            result = np.maximum(0., result)
            if result.shape[0] == 1:
                result = result[0]
        else:
            result = np.empty_like(x)
            result[:] = np.nan
            print(sol.message)

        return result

    def plot(self,
             kind: Literal['drift', 'kp', 'Mayo', 'triads'],
             show: Literal['auto', 'all', 'data', 'model'] = 'auto',
             M: Literal[1, 2] = 1,
             f0: Optional[Union[float, FloatVectorLike]] = None,
             T: Optional[float] = None,
             Tunit: Literal['C', 'K'] = 'K',
             title: Optional[str] = None,
             axes: Optional[Axes] = None,
             return_objects: bool = False
             ) -> Optional[tuple[Optional[Figure], Axes]]:
        r"""Generate a plot of instantaneous copolymer composition, monomer
        composition drift, or average propagation rate coefficient.

        Parameters
        ----------
        kind : Literal['drift', 'kp', 'Mayo', 'triads']
            Kind of plot to be generated.
        show : Literal['auto', 'all', 'data', 'model']
            What informatation is to be plotted.
        M : Literal[1, 2]
            Index of the monomer to be used in input argument `f0` and in
            output results. Specifically, if `M=i`, then `f0` stands for
            $f_i(0)$ and plots will be generated in terms of $f_i$ and $F_i$.
        f0 : float | FloatVectorLike | None
            Initial monomer composition, $f_i(0)$, as required for a monomer
            composition drift plot.
        T : float | None
            Temperature. Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.
        title : str | None
            Title of plot. If `None`, a default title with the monomer names
            will be used.
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
        check_in_set(M, {1, 2}, 'M')
        check_in_set(kind, {'Mayo', 'kp', 'drift', 'triads'}, 'kind')
        check_in_set(show, {'auto', 'all', 'data', 'model'}, 'show')

        label_model = None
        if axes is None:
            fig, ax = plt.subplots()
            if title is None:
                titles = {'Mayo': "Mayo-Lewis diagram",
                          'drift': "Monomer composition drift",
                          'kp': "Average propagation coefficient",
                          'triads': "Triad fractions"}
                title = titles[kind] + f" {self.M1}(1)-{self.M2}(2)"
            if title:
                fig.suptitle(title)
        else:
            ax = axes
            fig = None
            if self.name:
                label_model = self.name

        unit_range = (0., 1.)
        npoints = 1000
        Mname = self.M1 if M == 1 else self.M2
        ndataseries = 0

        if show == 'auto':
            if not self.data:
                show = 'model'
            else:
                show = 'all'

        if kind == 'Mayo':

            ax.set_xlabel(fr"$f_{M}$")
            ax.set_ylabel(fr"$F_{M}$")
            ax.set_xlim(*unit_range)
            ax.set_ylim(*unit_range)

            ax.plot(unit_range, unit_range, color='black', linewidth=0.5)

            if show in {'model', 'all'}:
                x = np.linspace(*unit_range, npoints, dtype=np.float64)
                y = self.F1(x)
                if M == 2:
                    x[:] = 1. - x
                    y[:] = 1. - y  # type: ignore
                ax.plot(x, y, label=label_model)

            if show in {'data', 'all'}:
                for ds in self.data:
                    if not isinstance(ds, MayoDataset):
                        continue
                    x = ds.getvar('f', Mname)
                    y = ds.getvar('F', Mname)
                    ndataseries += 1
                    ax.scatter(x, y,
                               label=ds.name if ds.name else None)

        elif kind == 'drift':

            ax.set_xlabel(r"Total molar monomer conversion, $x$")
            ax.set_ylabel(fr"$f_{M}$")
            ax.set_xlim(*unit_range)
            ax.set_ylim(*unit_range)

            if show == 'model':

                if f0 is None:
                    raise ValueError("`f0` is required for a `drift` plot.")
                else:
                    if isinstance(f0, (int, float)):
                        f0 = [f0]
                    f0 = np.array(f0, dtype=np.float64)
                    check_bounds(f0, *unit_range, 'f0')

                x = np.linspace(*unit_range, 1000)
                if M == 2:
                    f0[:] = 1. - f0
                y = self.drift(f0, x)
                if M == 2:
                    y[:] = 1. - y
                if y.ndim == 1:
                    y = y[np.newaxis, :]
                for i in range(y.shape[0]):
                    ax.plot(x, y[i, :], label=label_model)

            if show in {'data', 'all'}:

                for ds in self.data:
                    if not isinstance(ds, DriftDataset):
                        continue
                    x = ds.getvar('x')
                    y = ds.getvar('f', Mname)
                    ndataseries += 1
                    ax.scatter(x, y,
                               label=ds.name if ds.name else None)

                    if show == 'all':
                        x = np.linspace(*unit_range, npoints)
                        y = self.drift(ds.getvar('f', self.M1)[0], x)
                        if M == 2:
                            y[:] = 1. - y
                        ax.plot(x, y, label=label_model)

        elif kind == 'kp':

            ax.set_xlabel(fr"$f_{M}$")
            ax.set_ylabel(r"$\bar{k}_p$")
            ax.set_xlim(*unit_range)

            if show == 'model':

                if T is None:
                    raise ValueError("`T` is required for a `kp` plot.")

                x = np.linspace(*unit_range, npoints)
                y = self.kp(x, T, Tunit)
                if M == 2:
                    x[:] = 1. - x
                ax.plot(x, y, label=label_model)

            if show in {'data', 'all'}:

                for ds in self.data:
                    if not isinstance(ds, kpDataset):
                        continue
                    x = ds.getvar('f', Mname)
                    y = ds.getvar('kp')
                    ndataseries += 1
                    ax.scatter(x, y,
                               label=ds.name if ds.name else None)
                    if show == 'all':
                        x = np.linspace(*unit_range, npoints)
                        y = self.kp(x, ds.T, ds.Tunit)
                        if M == 2:
                            x[:] = 1. - x
                        ax.plot(x, y, label=label_model)

        ax.grid(True)

        if axes is not None or ndataseries:
            ax.legend(bbox_to_anchor=(1.05, 1.), loc="upper left")

        if return_objects:
            return (fig, ax)

    def add_data(self,
                 data: Union[CopoDataset, list[CopoDataset]]
                 ) -> None:
        r"""Add a copolymerization dataset for subsequent analysis.

        Parameters
        ----------
        data : Union[CopoDataset, list[CopoDataset]]
            Experimental dataset(s).
        """
        if not isinstance(data, list):
            data = [data]

        valid_monomers = {self.M1, self.M2}
        for ds in data:
            ds_monomers = {ds.M1, ds.M2}
            if ds_monomers != valid_monomers:
                raise ValueError(
                    f"Monomers defined in dataset `{ds.name}` are invalid: "
                    f"{ds_monomers}!={valid_monomers}")
            if ds not in set(self.data):
                self.data.append(ds)
            else:
                print(f"Warning: duplicate dataset '{ds.name}' was skipped.")


# %% Terminal model


class TerminalModel(CopoModel):
    r"""Terminal binary copolymerization model.

    According to this model, the reactivity of a macroradical depends
    uniquely on the nature of the _terminal_ repeating unit. A binary system
    is, thus, described by four propagation reactions:

    \begin{matrix}
    P^{\bullet}_1 + M_1 \overset{k_{11}}{\rightarrow} P^{\bullet}_1 \\
    P^{\bullet}_1 + M_2 \overset{k_{12}}{\rightarrow} P^{\bullet}_2 \\
    P^{\bullet}_2 + M_1 \overset{k_{21}}{\rightarrow} P^{\bullet}_1 \\
    P^{\bullet}_2 + M_2 \overset{k_{22}}{\rightarrow} P^{\bullet}_2
    \end{matrix}

    where $k_{ii}$ are the homopropagation rate coefficients and $k_{ij}$ are
    the cross-propagation coefficients. The two cross-propagation coefficients
    are specified via an equal number of reactivity ratios, defined as
    $r_1=k_{11}/k_{12}$ and $r_2=k_{22}/k_{21}$.

    Parameters
    ----------
    r1 : float
        Reactivity ratio of M1, $r_1=k_{11}/k_{12}$.
    r2 : float
        Reactivity ratio of M2, $r_2=k_{22}/k_{21}$.
    k1 : Arrhenius | None
        Homopropagation rate coefficient of M1, $k_1 \equiv k_{11}$.
    k2 : Arrhenius | None
        Homopropagation rate coefficient of M2, $k_2 \equiv k_{22}$.
    M1 : str
        Name of M1.
    M2 : str
        Name of M2.
    name : str
        Name.
    """

    r1: float
    r2: float

    _pnames = ('r1', 'r2')

    def __init__(self,
                 r1: float,
                 r2: float,
                 k1: Optional[Arrhenius] = None,
                 k2: Optional[Arrhenius] = None,
                 M1: str = 'M1',
                 M2: str = 'M2',
                 name: str = ''
                 ) -> None:

        check_bounds(r1, 0., np.inf, 'r1')
        check_bounds(r2, 0., np.inf, 'r2')

        # Perhaps this could be upgraded to exception, but I don't want to be
        # too restrictive (one does find literature data with (r1,r2)>1)
        if r1 > 1. and r2 > 1.:
            print(
                f"Warning: `r1`={r1} and `r2`={r2} are both greater than 1, "
                "which is deemed physically impossible.")

        self.r1 = r1
        self.r2 = r2
        super().__init__(k1, k2, M1, M2, name)

    @classmethod
    def from_Qe(cls,
                Qe1: tuple[float, float],
                Qe2: tuple[float, float],
                k1: Optional[Arrhenius] = None,
                k2: Optional[Arrhenius] = None,
                M1: str = 'M1',
                M2: str = 'M2',
                name: str = ''
                ) -> TerminalModel:
        r"""Construct `TerminalModel` from Q-e values.

        Alternative constructor that takes the $Q$-$e$ values of the monomers
        as primary input instead of the reactivity ratios.

        The conversion from Q-e to r is handled by
        [`convert_Qe_to_r`](convert_Qe_to_r.md).

        Parameters
        ----------
        Qe1 : tuple[float, float]
            Q-e values of M1.
        Qe2 : tuple[float, float]
            Q-e values of M2.
        k1 : Arrhenius | None
            Homopropagation rate coefficient of M1, $k_1 \equiv k_{11}$.
        k2 : Arrhenius | None
            Homopropagation rate coefficient of M2, $k_2 \equiv k_{22}$.
        M1 : str
            Name of M1.
        M2 : str
            Name of M2.
        name : str
            Name.
        """
        r = convert_Qe_to_r([Qe1, Qe2])
        r1 = r[0, 1]
        r2 = r[1, 0]
        return cls(r1, r2, k1, k2, M1, M2, name)

    @property
    def azeotrope(self) -> Optional[float]:
        r"""Calculate the azeotrope composition.

        An azeotrope (i.e., a point where $F_1=f_1$) only exists if both
        reactivity ratios are smaller than unity. In that case, the azeotrope
        composition is given by:

        $$ f_{1,azeo} = \frac{1 - r_2}{(2 - r_1 - r_2)} $$

        where $r_1$ and $r_2$ are the reactivity ratios.

        Returns
        -------
        float | None
            If an azeotrope exists, it returns its composition in terms of
            $f_1$.
        """
        r1 = self.r1
        r2 = self.r2
        if r1 < 1. and r2 < 1.:
            result = (1. - r2)/(2. - r1 - r2)
        else:
            result = None
        return result

    def ri(self,
           _
           ) -> tuple[Union[float, FloatArray], Union[float, FloatArray]]:
        return (self.r1, self.r2)

    def kii(self,
            _,
            T: float,
            Tunit: Literal['C', 'K'] = 'K'
            ) -> tuple[Union[float, FloatArray], Union[float, FloatArray]]:
        if self.k1 is None or self.k2 is None:
            raise ValueError(
                "To use this feature, `k1` and `k2` cannot be `None`.")
        return (self.k1(T, Tunit), self.k2(T, Tunit))

    def transitions(self,
                    f1: Union[float, FloatArrayLike]
                    ) -> dict[str, Union[float, FloatArray]]:
        r"""Calculate the instantaneous transition probabilities.

        For a binary system, the transition probabilities are given by:

        \begin{aligned}
            P_{ii} &= \frac{r_i f_i}{r_i f_i + (1 - f_i)} \\
            P_{ij} &= 1 - P_{ii}
        \end{aligned}

        where $i,j=1,2, i \neq j$.

        **References**

        * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 178.

        Parameters
        ----------
        f1 : float | FloatArrayLike
            Molar fraction of M1.

        Returns
        -------
        dict[str, float | FloatArray]
            Transition probabilities, {'11': $P_{11}$, '12': $P_{12}$, ... }.
        """

        if isinstance(f1, (list, tuple)):
            f1 = np.array(f1, dtype=np.float64)
        check_bounds(f1, 0., 1., 'f1')

        f2 = 1. - f1
        r1 = self.r1
        r2 = self.r2
        P11 = r1*f1/(r1*f1 + f2)
        P22 = r2*f2/(r2*f2 + f1)
        P12 = 1. - P11
        P21 = 1. - P22

        result = {'11': P11,
                  '12': P12,
                  '21': P21,
                  '22': P22}

        return result

    def triads(self,
               f1: Union[float, FloatArrayLike]
               ) -> dict[str, Union[float, FloatArray]]:
        r"""Calculate the instantaneous triad fractions.

        For a binary system, the triad fractions are given by:

        \begin{aligned}
            A_{iii} &= F_i P_{ii}^2 \\
            A_{iij} &= 2 F_i P_{ii} P_{ij} \\
            A_{jij} &= F_i P_{ij}^2
        \end{aligned}

        where $P_{ij}$ is the transition probability $i \rightarrow j$ and
        $F_i$ is the instantaneous copolymer composition.

        **References**

        * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 179.

        Parameters
        ----------
        f1 : float | FloatArrayLike
            Molar fraction of M1.

        Returns
        -------
        dict[str, float | FloatArray]
            Triad fractions,
            {'111': $A_{111}$, '112': $A_{112}$, '212': $A_{212}$, ... }.
        """

        P11, P12, P21, P22 = self.transitions(f1).values()

        F1 = P21/(P12 + P21)
        F2 = 1. - F1

        A111 = F1*P11**2
        A112 = F1*2*P11*P12
        A212 = F1*P12**2

        A222 = F2*P22**2
        A221 = F2*2*P22*P21
        A121 = F2*P21**2

        result = {'111': A111,
                  '112': A112,
                  '212': A212,
                  '222': A222,
                  '221': A221,
                  '121': A121}

        return result

    def sequence(self,
                 f1: Union[float, FloatArrayLike],
                 k: Optional[Union[int, IntArrayLike]] = None,
                 ) -> dict[str, Union[float, FloatArray]]:
        r"""Calculate the instantaneous sequence length probability or the
        number-average sequence length.

        For a binary system, the probability of finding $k$ consecutive units
        of monomer $i$ in a chain is:

        $$ S_{i,k} = (1 - P_{ii})P_{ii}^{k-1} $$

        and the corresponding number-average sequence length is:

        $$ \bar{S}_i = \sum_k k S_{i,k} = \frac{1}{1 - P_{ii}} $$

        where $P_{ii}$ is the transition probability $i \rightarrow i$, which
        is a function of the monomer composition and the reactivity ratios.

        **References**

        * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 177.

        Parameters
        ----------
        f1 : float | FloatArrayLike
            Molar fraction of M1.
        k : int | IntArrayLike | None
            Sequence length, i.e., number of consecutive units in a chain.
            If `None`, the number-average sequence length will be computed.

        Returns
        -------
        dict[str, float | FloatArray]
            If `k is None`, the number-average sequence lengths,
            {'1': $\bar{S}_1$, '2': $\bar{S}_2$}. Otherwise, the
            sequence probabilities, {'1': $S_{1,k}$, '2': $S_{2,k}$}.
        """

        P11, _, _, P22 = self.transitions(f1).values()

        if k is None:
            result = {str(i + 1): 1/(1 - P + eps)
                      for i, P in enumerate([P11, P22])}
        else:
            if isinstance(k, (list, tuple)):
                k = np.array(k, dtype=np.int32)
            result = {str(i + 1): (1. - P)*P**(k - 1)
                      for i, P in enumerate([P11, P22])}

        return result
