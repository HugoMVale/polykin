# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

from numpy import isclose

from polykin.kinetics.emulsion.entry import (
    kentry_collision,
    kentry_diffusion,
    kentry_diffusion_reversible,
)


def test_ke_collision():
    r = 200e-9  # m
    C = 1e-3  # m/s
    ke = kentry_collision(r, C)
    assert isclose(ke, 3.03e8, rtol=1e-2)


def test_ke_diffusion():
    r = 10e-9  # m
    Dw = 1e-9  # m²/s
    ke = kentry_diffusion(r, Dw)
    assert isclose(ke, 7.56e7, rtol=1e-2)


def test_ke_diffusion_reversible():
    r = 10e-9  # m
    Dw = 1e-9  # m²/s
    q = 10.0
    # fast reaction limit
    Dp = Dw
    k = 1e10  # s-1
    ke_irreversible = kentry_diffusion(r, Dw)
    ke_reversible = kentry_diffusion_reversible(r, Dw, Dp, q, k)
    assert isclose(ke_reversible, ke_irreversible, rtol=1e-2)
