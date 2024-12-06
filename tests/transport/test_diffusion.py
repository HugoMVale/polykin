# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import isclose

from polykin.transport.diffusion import (diffusivity_composite,
                                         profile_constc_semiinf,
                                         profile_constc_sheet,
                                         profile_constc_sphere,
                                         uptake_constc_sheet,
                                         uptake_constc_sphere,
                                         uptake_convection_sheet,
                                         uptake_convection_sphere)


def test_profile_constc_semiinf():
    "Seeder and Henley, p. 120"
    q = profile_constc_semiinf(t=2.09*3600, x=1, D=1e-5)
    assert isclose(q, 0.01, rtol=1e-2)


def test_profile_constc_sheet():
    # For short times, it should be almost as infinite medium
    D = 1e-9
    a = 1.
    x = 0.001
    t = 1e3
    q1 = profile_constc_semiinf(t, x, D)
    q2 = profile_constc_sheet(D*t/a**2, (a-x)/a)
    assert isclose(q1, q2, rtol=1e-8)
    # Long times, read from Fig 4.1 of Crank
    q = profile_constc_sheet(1, 0.1)
    assert isclose(q, 0.9, atol=0.01)


def test_profile_constc_sphere():
    # For short times, and small curvature it should be almost as infinite medium
    D = 1e-9
    a = 1.
    x = 0.0001
    t = 1e3
    q1 = profile_constc_semiinf(t, x, D)
    q2 = profile_constc_sphere(D*t/a**2, (a-x)/a)
    assert isclose(q1, q2, rtol=1e-3)
    # Long times, read from Fig 6.1 of Crank
    assert isclose(profile_constc_sphere(0.3, 0.1), 0.9, atol=0.01)
    # Particular case of r=0
    assert isclose(profile_constc_sphere(0.3, 0), 0.9, atol=0.01)
    assert isclose(profile_constc_sphere(0.04, 0), 0.01, atol=0.003)


def test_uptake_constc_sheet():
    assert isclose(uptake_constc_sheet(0), 0)
    assert isclose(uptake_constc_sheet(1e5), 1)
    assert isclose(uptake_constc_sheet(0.25 + 1e-7),
                   uptake_constc_sheet(0.25 - 1e-7))
    # "Seeder and Henley, p. 123"
    assert isclose(uptake_constc_sheet(0.56), 0.8, atol=0.01)
    assert isclose(uptake_constc_sheet(0.2), 0.5, atol=0.02)


def test_uptake_constc_sphere():
    assert isclose(uptake_constc_sphere(0), 0)
    assert isclose(uptake_constc_sphere(1e5), 1)
    assert isclose(uptake_constc_sphere(0.25 - 1e-7),
                   uptake_constc_sphere(0.25 + 1e-7))
    # "Seeder and Henley, p. 123"
    assert isclose(1-uptake_constc_sphere(0.65), 0.001, rtol=1e-2)
    assert isclose(1-uptake_constc_sphere(0.11), 0.2, atol=0.01)


def test_diffusivity_composite():
    for Dd, Dc in [(1, 2), (2, 1)]:
        assert isclose(diffusivity_composite(Dd, Dc, fd=0), Dc)
        assert isclose(diffusivity_composite(Dd, Dc, fd=1), Dd)
        assert isclose(diffusivity_composite(Dd, Dc, fd=0.5),
                       (Dd+Dc)/2, rtol=0.1)


def test_uptake_convection_sheet():
    assert isclose(uptake_convection_sheet(0, 1), 0.)
    assert isclose(uptake_convection_sheet(1e2, 1), 1.)
    assert isclose(uptake_convection_sheet(2**2, 0.5), 0.80, atol=0.02)
    assert isclose(uptake_convection_sheet(2**2, 0.1), 0.32, atol=0.02)
    assert isclose(uptake_convection_sheet(0.25, 1e6),
                   uptake_constc_sheet(0.25))


def test_uptake_convection_sphere():
    assert isclose(uptake_convection_sphere(0, 1), 0.)
    assert isclose(uptake_convection_sphere(1e2, 1), 1.)
    assert isclose(uptake_convection_sphere(1**2, 0.2), 0.45, atol=0.02)
    assert isclose(uptake_convection_sphere(1**2, 0.5), 0.75, atol=0.02)
    assert isclose(uptake_convection_sphere(0.25, 1e6),
                   uptake_constc_sphere(0.25))
