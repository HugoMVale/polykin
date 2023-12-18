# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from .base import EoS

from scipy.constants import R

__all__ = ['IdealGas']


class IdealGas(EoS):

    def Z(self, T, P, y):
        return 1.

    def P(self, T, v, y):
        return R*T/v

    def phi(self, T, P, y):
        return 1.

    def Ares(self, T, V, n):
        return 0.
