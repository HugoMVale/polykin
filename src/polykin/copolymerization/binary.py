# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.utils import FloatVectorLike, FloatVector, ShapeError, eps
from polykin import utils
from polykin.base import Base

from dataclasses import dataclass
from operator import itemgetter
import numpy as np
from typing import Union, Optional, Literal, Any
from scipy.stats import linregress
from scipy.stats.distributions import t
from scipy.optimize import curve_fit
from scipy import odr
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse, Patch
from matplotlib.figure import Figure

__all__ = ['CopoData', 'CopoFitResult', 'TerminalModel']

# %% Dataclasses


@dataclass(frozen=True)
class CopoData():
    M1: str
    M2: str
    f1: FloatVectorLike
    F1: FloatVectorLike
    sigma_f: Optional[Union[float, FloatVectorLike]] = 1e-3
    sigma_F: Optional[Union[float, FloatVectorLike]] = 5e-2
    name: Optional[str] = ''
    reference: Optional[str] = ''

    def __post_init__(self):
        if not self.M1 or not self.M2 or self.M1.lower() == self.M2.lower():
            raise ValueError(
                "`M1` and `M2` must be non-empty and different.")
        for attr in ['f1', 'F1', 'sigma_f', 'sigma_F']:
            x = getattr(self, attr)
            if isinstance(x, list):
                x = np.array(x, dtype=np.float64)
                object.__setattr__(self, attr, x)
            utils.check_bounds(x, 0., 1., attr)
        if len(self.f1) != len(self.F1):
            raise ShapeError(
                "`f1` and `F1` must be vectors of the same length.")
        for attr in ['sigma_f', 'sigma_F']:
            x = getattr(self, attr)
            if isinstance(x, np.ndarray) and (len(x) != len(self.f1)):
                raise ShapeError(
                    f"`{attr}` must have the same length as `f1` and `F1`.")

    def __repr__(self):
        return \
            f"M1:        {self.M1}\n" \
            f"M2:        {self.M2}\n" \
            f"f1:        {self.f1}\n" \
            f"F1:        {self.F1}\n" \
            f"sigma_f:   {self.sigma_f}\n" \
            f"sigma_F:   {self.sigma_F}\n" \
            f"name:      {self.name}\n" \
            f"reference: {self.reference}"


@dataclass(frozen=True)
class CopoFitResult():
    """Something"""
    M1: str
    M2: str
    r1: Optional[float] = None
    r2: Optional[float] = None
    sigma_r1: Optional[float] = None
    sigma_r2: Optional[float] = None
    error95_r1: Optional[float] = None
    error95_r2: Optional[float] = None
    cov: Optional[Any] = None
    method: str = ''

    def __repr__(self):
        s1 = \
            f"method:     {self.method}\n" \
            f"M1:         {self.M1}\n" \
            f"M2:         {self.M2}\n" \
            f"r1:         {self.r1:.2E}\n" \
            f"r2:         {self.r2:.2E}\n"
        if self.sigma_r1 is not None:
            s2 = \
                f"sigma_r1:   {self.sigma_r1:.2E}\n" \
                f"sigma_r2:   {self.sigma_r2:.2E}\n" \
                f"error95_r1: {self.error95_r1:.2E}\n" \
                f"error95_r2: {self.error95_r2:.2E}\n" \
                f"cov:        {self.cov}\n"
        else:
            s2 = ""
        return s1 + s2

# %% Models


class CopoModel(Base):
    pass


class TerminalModel(CopoModel):

    def __init__(self,
                 r1: float,
                 r2: float,
                 M1: str = 'M1',
                 M2: str = 'M2',
                 name: str = ''
                 ) -> None:

        utils.check_bounds(r1, 0., np.inf, 'r1')
        utils.check_bounds(r2, 0., np.inf, 'r2')
        self.r1 = r1
        self.r2 = r2

        # Perhaps this could be upgraded to exception, but I don't want to be
        # too restrictive (one does find literature data with (r1,r2)>1)
        if r1 > 1. and r2 > 1.:
            print(
                f"Warning: `r1`={r1} and `r2`={r2} are both greater than 1, which is deemed physically impossible.")

        if M1 and M2 and M1.lower() != M2.lower():
            self.M1 = M1
            self.M2 = M2
        else:
            raise ValueError("`M1` and `M2` must be non-empty and different.")
        pass

        self.name = name

    @property
    def azeo(self
             ) -> Union[float, None]:
        r1 = self.r1
        r2 = self.r2
        if r1 < 1. and r2 < 1.:
            result = (1 - r2)/(2 - r1 - r2)
        else:
            result = None
        return result

    def F1inst(self,
               f1: Union[float, FloatVectorLike],
               check_bounds: bool = True
               ) -> Union[float, FloatVector]:
        if isinstance(f1, list):
            f1 = np.array(f1, dtype=np.float64)
        if check_bounds:
            utils.check_bounds(f1, 0., 1., 'f1')
        f2 = 1 - f1
        r1 = self.r1
        r2 = self.r2
        return (r1*f1**2 + f1*f2)/(r1*f1**2 + 2*f1*f2 + r2*f2**2)

    def drift(self,
              f10: Union[float, FloatVectorLike],
              x: Union[float, FloatVectorLike]
              ) -> FloatVector:

        if isinstance(f10, (int, float)):
            f10 = [f10]
        if isinstance(x, (int, float)):
            x = [x]

        def df1dx(x_, f1_):
            return (f1_ - self.F1inst(f1_, False))/(1 - x_ + eps)

        sol = solve_ivp(df1dx,
                        (0, max(x)),
                        f10,
                        method='LSODA',
                        t_eval=x,
                        vectorized=True,
                        rtol=1e-4)
        if sol.success:
            result = sol.y
        else:
            result = np.empty_like(x)
            result[:] = np.nan
            print(sol.message)

        return result

    def plot(self,
             M: Literal[1, 2] = 1,
             f0: Union[float, FloatVectorLike, None] = None,
             title: Union[str, None] = None,
             axes: Union[plt.Axes, None] = None,
             ) -> Union[Figure, None]:

        utils.check_in_set(M, {1, 2}, 'M')

        if f0 is not None:
            if isinstance(f0, (int, float)):
                f0 = [f0]
            f0 = np.array(f0, dtype=np.float64)
            utils.check_bounds(f0, 0., 1., 'f0')

        if axes is None:
            fig, ax = plt.subplots()
            if title is None:
                if f0 is None:
                    title = "Mayo plot"
                else:
                    title = "Composition drift"
                title += f" {self.M1}(1)-{self.M2}(2)"
            if title:
                fig.suptitle(title)
            label = None
        else:
            ax = axes
            fig = None
            if self.name:
                label = self.name

        if f0 is None:
            ax.set_xlabel(fr"$f_{M}$")
            ax.set_ylabel(fr"$F_{M}$")

            ax.plot((0., 1.), (0., 1.), color='black', linewidth=0.5)

            x = np.linspace(0, 1, 1000)
            y = self.F1inst(x)
            if M == 2:
                x[:] = 1 - x
                y[:] = 1 - y
            ax.plot(x, y, label=label)

        else:
            ax.set_xlabel("Total molar monomer conversion, " + r"$x$")
            ax.set_ylabel(fr"$f_{M}$")

            x = np.linspace(0, 1, 200)

            y = self.drift(f0, x)
            for i in range(len(f0)):
                label = 'x'
                ax.plot(x, y[i, :], label=label)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)

        if axes is not None:
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        return fig


# %% Analysis


class CopoAnalysis():

    def __init__(self,
                 data: Union[CopoData, list[CopoData]],
                 M1: Union[str, None] = None,
                 M2: Union[str, None] = None
                 ) -> None:

        # Recast data input as list and check content
        if not isinstance(data, list):
            data = [data]
        self.data = utils.check_type(data, CopoData, 'data',
                                     check_inside=True)

        # Define M1, M2
        if M1 and M2:
            if M1.lower() != M2.lower():
                self.M1 = M1
                self.M2 = M2
            else:
                raise ValueError(
                    f"M1='{M1}' and M2='{M2}' must be different monomers.")
        elif not (M1 or M2):
            self.M1 = data[0].M1
            self.M2 = data[0].M2
        else:
            raise ValueError(
                "`M1` and `M2` must be both defined or both undefined.")

        # Check monomer consistency in datasets
        valid_monomers = {self.M1, self.M2}
        for ds in self.data:
            ds_monomers = {ds.M1, ds.M2}
            if valid_monomers != ds_monomers:
                raise ValueError(
                    f"Monomers defined in dataset `{ds.name}` are invalid: {ds_monomers}!={valid_monomers}")

        self._data_fit = {}

    def plot(self,
             M: Literal[1, 2] = 1,
             title: Union[str, None] = None,
             axes: Union[plt.Axes, None] = None,
             show: Literal['data', 'fit', 'all'] = 'all'
             ) -> Union[Figure, None]:

        if axes is None:
            fig, ax = plt.subplots()
            if title is None:
                title = f"Mayo plot {self.M1}(1)-{self.M2}(2)"
            if title:
                fig.suptitle(title)
        else:
            ax = axes
            fig = None

        if M == 1:
            Mx = self.M1
        elif M == 2:
            Mx = self.M2
        else:
            raise ValueError("Monomer index `M` must be 1 or 2.")

        ax.set_xlabel(fr"$f_{M}$")
        ax.set_ylabel(fr"$F_{M}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.plot((0, 1), (0, 1), color='black', linewidth=0.5, label=None)

        if show in {'data', 'all'}:
            for ds in self.data:
                x = ds.f1
                y = ds.F1
                if Mx == ds.M2:
                    x[:] = 1 - x
                    y[:] = 1 - y
                ax.scatter(x, y, label=ds.name)
        if show in {'fit', 'all'}:
            x = np.linspace(0, 1, 100
                            )
            pass

        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        return fig

    def fit(self,
            method: Literal['FR', 'NLLS', 'ODR'] = 'NLLS',
            alpha: float = 0.05,
            plot: bool = True
            ) -> CopoFitResult:

        method_names = {'FR': 'Finemann-Ross',
                        'NLLS': 'Non-linear least squares',
                        'ODR': 'Orthogonal distance regression'}

        # Concatenate all datasets and save to cache
        if not self._data_fit:
            f1 = np.array([])
            F1 = np.array([])
            sigma_f = np.array([])
            sigma_F = np.array([])
            for ds in self.data:
                ds_f1 = ds.f1
                ds_F1 = ds.F1
                npoints = len(ds_f1)
                if ds.M1 == self.M2 and ds.M2 == self.M1:
                    ds_f1 = 1 - ds_f1
                    ds_F1 = 1 - ds_F1
                f1 = np.concatenate([f1, ds_f1])
                F1 = np.concatenate([F1, ds_F1])
                if isinstance(ds.sigma_f, float):
                    ds_sigma_f = np.full(npoints, ds.sigma_f)
                else:
                    ds_sigma_f = ds.sigma_f
                if isinstance(ds.sigma_F, float):
                    ds_sigma_F = np.full(npoints, ds.sigma_F)
                else:
                    ds_sigma_F = ds.sigma_F
                sigma_f = np.concatenate([sigma_f, ds_sigma_f])
                sigma_F = np.concatenate([sigma_F, ds_sigma_F])

            # Remove invalid f, F values
            idx_valid = np.logical_and.reduce((f1 > 0, f1 < 1, F1 > 0, F1 < 1))
            f1 = f1[idx_valid]
            F1 = F1[idx_valid]
            sigma_f = sigma_f[idx_valid]
            sigma_F = sigma_F[idx_valid]

            # Store in cache
            self._data_fit.update({'f1': f1,
                                   'F1': F1,
                                   'sigma_f': sigma_f,
                                   'sigma_F': sigma_F})
        else:
            f1, F1, sigma_f, sigma_F = \
                itemgetter('f1', 'F1', 'sigma_f', 'sigma_F')(self._data_fit)

        # Finemann-Ross (either for itself or as initial guess for other methods)
        x, y = f1/(1 - f1), F1/(1 - F1)
        x, y = -y/x**2, (y - 1)/x
        solution = linregress(x, y)
        _r1, _r2 = solution.intercept, solution.slope  # type: ignore

        r1 = None
        r2 = None
        sigma_r1 = None
        sigma_r2 = None
        cov = None
        error95_r1 = None
        error95_r2 = None

        if method == 'FR':
            r1 = _r1
            r2 = _r2

        elif method == 'NLLS':
            solution = curve_fit(F1_inst,
                                 xdata=f1,
                                 ydata=F1,
                                 p0=(_r1, _r2),
                                 sigma=sigma_F,
                                 absolute_sigma=True,
                                 bounds=(0, np.inf),
                                 full_output=True)
            if solution[4]:
                r1, r2 = solution[0]
                cov = solution[1]
                # This next part is to be checked
                sigma_r1, sigma_r2 = np.sqrt(np.diag(cov))
                tval = t.ppf(1 - alpha/2, max(0, f1.size - cov.shape[0]))
                error95_r1 = sigma_r1*tval
                error95_r2 = sigma_r2*tval
            else:
                print("Fit error: ", solution[3])

        elif method == 'ODR':
            odr_Model = odr.Model(lambda beta, x: F1_inst(x, *beta))
            odr_Data = odr.RealData(x=f1, y=F1, sx=sigma_f, sy=sigma_F)
            odr_ODR = odr.ODR(odr_Data, odr_Model, beta0=(_r1, _r2))
            solution = odr_ODR.run()
            r1, r2 = solution.beta
            cov = np.array(solution.cov_beta)  # !!! not sure
            # This next part is to be checked + finished
            sigma_r1, sigma_r2 = solution.sd_beta
            error95_r1 = sigma_r1
            error95_r2 = sigma_r2

        else:
            utils.check_in_set(method, set(method_names.keys()), 'method')

        # Pack results into object
        result = CopoFitResult(M1=self.M1,
                               M2=self.M2,
                               r1=r1,
                               r2=r2,
                               sigma_r1=sigma_r1,
                               sigma_r2=sigma_r2,
                               error95_r1=error95_r1,
                               error95_r2=error95_r2,
                               cov=cov,
                               method=method_names[method])
        if r1 is not None and r2 is not None and cov is not None:
            draw_jcr(r1, r2, cov, alpha)
        return result


def draw_jcr(r1: float,
             r2: float,
             cov: np.ndarray,
             alpha: float = 0.05):

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$r_1$")
    ax.set_ylabel(r"$r_2$")
    ax.scatter(r1, r2, c='black', s=5)
    confidence_ellipse((r1, r2), cov, ax)
    return


def confidence_ellipse(center: tuple[float, float],
                       cov: np.ndarray,
                       ax: plt.Axes,
                       nstd: float = 1.96
                       ) -> Patch:

    pearson = cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])
    radius_x = np.sqrt(1 + pearson)
    radius_y = np.sqrt(1 - pearson)
    scale_x, scale_y = np.sqrt(np.diag(cov))*nstd

    ellipse = Ellipse((0, 0),
                      width=2*radius_x,
                      height=2*radius_y,
                      facecolor='none',
                      edgecolor='black')

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(*center)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def F1_inst(f1, r1, r2):
    f2 = (1 - f1)
    return (r1*f1**2 + f1*f2)/(r1*f1**2 + 2*f1*f2 + r2*f2**2)
