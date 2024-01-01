# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse, Patch
from scipy import odr
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.stats.distributions import t

from polykin import utils
from polykin.types import (FloatOrVector, FloatOrVectorLike, FloatVector,
                           FloatVectorLike)
from polykin.utils import ShapeError, check_shapes
from polykin.math import convert_FloatOrVectorLike_to_FloatOrVector

__all__ = ['CopoFitResult']


# %% CopoFitResult


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

# %% Aux functions


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

# %%


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

# %% Fit functions


def fit_Finemann_Ross(f1: FloatVector,
                      F1: FloatVector
                      ) -> tuple[float, float]:
    # Finemann-Ross (either for itself or as initial guess for other methods)
    x, y = f1/(1 - f1), F1/(1 - F1)
    x, y = -y/x**2, (y - 1)/x
    solution = linregress(x, y)
    r1, r2 = solution.intercept, solution.slope
    return (r1, r2)
