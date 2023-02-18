# from dataclasses import dataclass

from numba import njit
import numpy as np
import pydantic
from typing import Union, Optional, Any
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy import odr

# need to improve this...
Vector = Union[list[float], Any]


class Monomer(pydantic.BaseModel):
    uid: str
    name: str


class CopoData(pydantic.BaseModel):
    monomers: list[Monomer]
    f: Vector
    F: Vector
    sigmaf: Optional[float] = 1e-3
    sigmaF: Optional[float] = 1.
    reference: Optional[str] = ''


# %% Functions

def fit_ratios(data: Union[CopoData, list[CopoData]], method='FR'):

    # Recast input as list
    if not isinstance(data, list):
        data = [data]

    # Join all datasets
    f = np.asarray([])
    F = np.asarray([])
    sigmaf = np.asarray([])
    sigmaF = np.asarray([])
    for ds in data:
        f = np.concatenate([f, ds.f])
        F = np.concatenate([F, ds.F])
        sigmaf = np.concatenate([sigmaf, np.full(len(ds.f), ds.sigmaf)])
        sigmaF = np.concatenate([sigmaF, np.full(len(ds.F), ds.sigmaF)])

    # Remove 0s and 1s
    idx_valid = np.logical_and.reduce((f > 0, f < 1, F > 0, F < 1))
    f = f[idx_valid]
    F = F[idx_valid]
    sigmaf = sigmaf[idx_valid]
    sigmaF = sigmaF[idx_valid]

    # Init output
    result = {'r1': None,
              'r2': None,
              'sd_r1': None,
              'sd_r2': None,
              'cov': None,
              'method': ''}

    # Finemann-Ross (either for itself or as initializer)
    x, y = f/(1 - f), F/(1 - F)
    x, y = -y/x**2, (y - 1)/x
    solution = linregress(x, y)
    r1, r2 = solution.intercept, solution.slope  # type: ignore

    if method == 'FR':
        result['r1'] = r1
        result['r2'] = r2
        result['method'] = method
    elif method == 'NLLS':
        solution = curve_fit(f_to_F,
                             xdata=f,
                             ydata=F,
                             p0=(r1, r2),
                             sigma=sigmaF,
                             absolute_sigma=False,
                             bounds=(0, np.inf),
                             full_output=True)
        if solution[4] > 0:
            result['r1'] = solution[0][0]
            result['r2'] = solution[0][1]
            pcov = solution[1]
            perr = np.sqrt(np.diag(pcov))
            result['sd_r1'] = perr[0]
            result['sd_r2'] = perr[1]
            result['cov'] = pcov
        else:
            print("Fit error: ", solution[3])
    elif method == 'ODR':
        odr_Model = odr.Model(lambda beta, x: f_to_F(x, *beta))
        odr_Data = odr.RealData(x=f, y=F, sx=sigmaf, sy=sigmaF)
        odr_ODR = odr.ODR(odr_Data, odr_Model, beta0=(r1, r2))
        solution = odr_ODR.run()
        solution.pprint()

    return result


@njit(fastmath=True)
def f_to_F(f1, r1, r2):
    f2 = (1 - f1)
    return (r1*f1**2 + f1*f2)/(r1*f1**2 + 2*f1*f2 + r2*f2**2)
