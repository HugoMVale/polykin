from dataclasses import dataclass
from polykin.utils import FloatVectorLike
from polykin import utils

import numpy as np
from typing import Union, Optional, Literal, Any
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy import odr


@dataclass
class Monomer():
    uid: str
    name: str


@dataclass
class CopoData():
    M1: Monomer
    M2: Monomer
    f1: FloatVectorLike
    F1: FloatVectorLike
    sigmaf: Optional[float] = 1e-3
    sigmaF: Optional[float] = 5e-2
    name: Optional[str] = ''
    reference: Optional[str] = ''


@dataclass
class CopoFitResult():
    M1: Monomer
    M2: Monomer
    r1: Optional[float] = None
    r2: Optional[float] = None
    sdev_r1: Optional[float] = None
    sdev_r2: Optional[float] = None
    cov: Optional[Any] = None
    method: str = ''

# %% Functions


def fit_ratios(data: Union[CopoData, list[CopoData]],
               method: Literal['FR', 'NLLS', 'ODR'] = 'FR'
               ) -> CopoFitResult:

    method_names = {'FR': 'Finemann-Ross',
                    'NLLS': 'Non-linear least squares',
                    'ODR': 'Orthogonal distance regression'}

    # Recast input as list
    if not isinstance(data, list):
        data = [data]

    # Define M1,M2 based on first dataset
    M1 = data[0].M1
    M2 = data[0].M2

    # Join all datasets and check monomer consistency
    f = np.asarray([])
    F = np.asarray([])
    sigmaf = np.asarray([])
    sigmaF = np.asarray([])
    for ds in data:
        ftemp = np.asarray(ds.f1)
        Ftemp = np.asarray(ds.F1)
        if ds.M1 == M1 and ds.M2 == M2:
            pass
        elif ds.M1 == M2 and ds.M2 == M1:
            ftemp = 1 - ftemp
            Ftemp = 1 - Ftemp
        else:
            raise ValueError(f"Dataset {ds.name} contains invalid monomers.")
        if len(ftemp) != len(Ftemp):
            raise ValueError(
                f"Dataset {ds.name} has inconsistent (f1, F1) data.")
        f = np.concatenate([f, ftemp])
        F = np.concatenate([F, Ftemp])
        sigmaf = np.concatenate([sigmaf, np.full(len(ds.f1), ds.sigmaf)])
        sigmaF = np.concatenate([sigmaF, np.full(len(ds.f1), ds.sigmaF)])

    # Remove 0s and 1s
    idx_valid = np.logical_and.reduce((f > 0, f < 1, F > 0, F < 1))
    f = f[idx_valid]
    F = F[idx_valid]
    sigmaf = sigmaf[idx_valid]
    sigmaF = sigmaF[idx_valid]

    # Init output
    result = CopoFitResult(M1=M1, M2=M2)

    # Finemann-Ross (either for itself or as initializer)
    x, y = f/(1 - f), F/(1 - F)
    x, y = -y/x**2, (y - 1)/x
    solution = linregress(x, y)
    r1, r2 = solution.intercept, solution.slope  # type: ignore

    if method == 'FR':
        result.r1 = r1
        result.r2 = r2
    elif method == 'NLLS':
        solution = curve_fit(F1_inst,
                             xdata=f,
                             ydata=F,
                             p0=(r1, r2),
                             sigma=sigmaF,
                             absolute_sigma=False,
                             bounds=(0, np.inf),
                             full_output=True)
        if solution[4]:
            result.r1, result.r2 = solution[0]
            pcov = solution[1]
            result.sdev_r1, result.sdev_r2 = np.sqrt(np.diag(pcov))
            result.cov = pcov  # !!! not sure
        else:
            print("Fit error: ", solution[3])
    elif method == 'ODR':
        odr_Model = odr.Model(lambda beta, x: F1_inst(x, *beta))
        odr_Data = odr.RealData(x=f, y=F, sx=sigmaf, sy=sigmaF)
        odr_ODR = odr.ODR(odr_Data, odr_Model, beta0=(r1, r2))
        solution = odr_ODR.run()
        result.r1, result.r2 = solution.beta
        result.sdev_r1, result.sdev_r2 = solution.sd_beta
        result.cov = solution.cov_beta  # !!! not sure
    else:
        utils.check_in_set(method, set(method_names.keys()), 'method')

    result.method = method_names[method]

    return result

# Not sure how the cov arrays of NLLS and ODR relate to each other


def F1_inst(f1, r1, r2):
    f2 = (1 - f1)
    return (r1*f1**2 + f1*f2)/(r1*f1**2 + 2*f1*f2 + r2*f2**2)
