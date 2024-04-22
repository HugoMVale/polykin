# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from scipy.optimize import minimize

from polykin.copolymerization.binary import (inst_copolymer_binary,
                                             monomer_drift_binary)
from polykin.utils.exceptions import FitError
from polykin.utils.types import FloatVector


@dataclass(frozen=True)
class CopoDataset_Ff():
    """Dataclass for instantaneous copolymerization data of the form F(f)."""
    name: str
    f1: FloatVector
    F1: FloatVector
    scale_f: FloatVector
    scale_F: FloatVector
    weight: float = 1.


@dataclass(frozen=True)
class CopoDataset_fx():
    """Dataclass for drift copolymerization data of the form f1(x)."""
    name: str
    f10: float
    x: FloatVector
    f1: FloatVector
    scale_x: FloatVector
    scale_f: FloatVector
    weight: float = 1.


@dataclass(frozen=True)
class CopoDataset_Fx():
    """Dataclass for drift copolymerization data of the form F1(x)."""
    name: str
    f10: float
    x: FloatVector
    F1: FloatVector
    scale_x: FloatVector
    scale_F: FloatVector
    weight: float = 1.


def fit_copo(data_Ff: list[CopoDataset_Ff] = [],
             data_fx: list[CopoDataset_fx] = [],
             data_Fx: list[CopoDataset_Fx] = [],
             initial_guess: tuple[float, float] = (1.0, 1.0),
             alpha: float = 0.05,
             method='Powell',
             plots: bool = False
             ):
    "Warning: This is work in progress!"

    def sse(r: tuple[float, float]) -> float:
        "Total sum of squared errors."
        res = 0.
        # F(f) datasets
        for ds in data_Ff:
            F1_est = inst_copolymer_binary(ds.f1, *r)
            ey = (ds.F1 - F1_est)/ds.scale_F
            res += ds.weight*dot(ey, ey)

        # f(x) datasets
        for ds in data_fx:
            f1_est = monomer_drift_binary(ds.f10, ds.x, *r)
            ey = (ds.f1 - f1_est)/ds.scale_f
            res += ds.weight*dot(ey, ey)

        # f(x) datasets
        for ds in data_Fx:
            f1_est = monomer_drift_binary(ds.f10, ds.x, *r)
            F1_est = inst_copolymer_binary(f1_est, *r)
            ey = (ds.F1 - F1_est)/ds.scale_F
            res += ds.weight*dot(ey, ey)
        return res

    # Parameter estimation
    sol = minimize(sse,
                   x0=initial_guess,
                   bounds=((1e-3, 1e2), (1e-3, 1e2)),
                   method=method,
                   options={'maxiter': 200})
    if sol.success:
        ropt = sol.x
        # get cov as well - need to learn the math
    else:
        raise FitError(sol.message)

    if plots:
        # Plot F(f) data
        if data_Ff:
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$f_1$")
            ax.set_ylabel(r"$F_1$")
            ax.set_xlim(0., 1.)
            ax.set_ylim(0., 1.)
            for ds in data_Ff:
                ax.scatter(ds.f1, ds.F1, label=ds.name)
            x = np.linspace(0., 0.999, 200)
            y = inst_copolymer_binary(x, *ropt)
            ax.plot(x, y)
            ax.legend(loc="best")

        # Plot f(x) data
        if data_fx:
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$f_1$")
            ax.set_xlim(0., 1.)
            ax.set_ylim(0., 1.)
            x = np.linspace(0., 0.999, 200)
            for ds in data_fx:
                ax.scatter(ds.x, ds.f1, label=ds.name)
                f1_est = monomer_drift_binary(ds.f10, x, *ropt)
                ax.plot(x, f1_est)
            ax.legend(loc="best")

        # Plot F(x) data
        if data_Fx:
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$F_1$")
            ax.set_xlim(0., 1.)
            ax.set_ylim(0., 1.)
            x = np.linspace(0., 0.999, 200)
            for ds in data_Fx:
                ax.scatter(ds.x, ds.F1, label=ds.name)
                f1_est = monomer_drift_binary(ds.f10, x, *ropt)
                F1_est = inst_copolymer_binary(f1_est, *ropt)
                ax.plot(x, F1_est)
            ax.legend(loc="best")

    return ropt
