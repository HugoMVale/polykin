# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import isclose

from polykin.transport.diffusion import (profile_semiinf, profile_sheet,
                                         profile_sphere, uptake_sheet,
                                         uptake_sphere, diffusivity_composite)


def test_profile_semiinf():
    "Seeder and Henley, p. 120"
    q = profile_semiinf(t=2.09*3600, x=1, D=1e-5)
    assert isclose(q, 0.01, rtol=1e-2)


def test_profile_sheet():
    # For short times, it should be almost as infinite medium
    D = 1e-9
    a = 1.
    x = 0.001
    t = 1e3
    q1 = profile_semiinf(t, x, D)
    q2 = profile_sheet(t, a-x, a, D)
    assert isclose(q1, q2, rtol=1e-8)
    # Long times, read from Fig 4.1 of Crank
    q = profile_sheet(t=1, x=0.1, a=1, D=1)
    assert isclose(q, 0.9, atol=0.01)


def test_profile_sphere():
    # For short times, and small curvature it should be almost as infinite medium
    D = 1e-9
    a = 1.
    x = 0.0001
    t = 1e3
    q1 = profile_semiinf(t, x, D)
    q2 = profile_sphere(t, a-x, a, D)
    assert isclose(q1, q2, rtol=1e-3)
    # Long times, read from Fig 6.1 of Crank
    assert isclose(profile_sphere(t=0.3, r=0.1, a=1, D=1), 0.9, atol=0.01)
    # Particular case of r=0
    assert isclose(profile_sphere(t=0.3, r=0, a=1, D=1), 0.9, atol=0.01)
    assert isclose(profile_sphere(t=0.04, r=0, a=1, D=1), 0.01, atol=0.003)


def test_uptake_sheet():
    # assert isclose(uptake_sheet(t=0, a=1, D=1), 0)
    # assert isclose(uptake_sheet(t=1e5, a=1, D=1), 1)
    assert isclose(uptake_sheet(t=1, a=2+1e-5, D=1),
                   uptake_sheet(t=1, a=2-1e-5, D=1))
    # "Seeder and Henley, p. 123"
    assert isclose(uptake_sheet(t=0.56, a=1, D=1), 0.8, atol=0.01)
    assert isclose(uptake_sheet(t=0.2, a=1, D=1), 0.5, atol=0.02)


def test_uptake_sphere():
    assert isclose(uptake_sphere(t=0, a=1, D=1), 0)
    assert isclose(uptake_sphere(t=1e5, a=1, D=1), 1)
    assert isclose(uptake_sphere(t=1, a=2+1e-5, D=1),
                   uptake_sphere(t=1, a=2-1e-5, D=1))
    # "Seeder and Henley, p. 123"
    assert isclose(1-uptake_sphere(t=0.65, a=1, D=1), 0.001, rtol=1e-2)
    assert isclose(1-uptake_sphere(t=0.11, a=1, D=1), 0.2, atol=0.01)


def test_diffusivity_composite():
    for Dd, Dc in [(1, 2), (2, 1)]:
        assert isclose(diffusivity_composite(Dd, Dc, fd=0), Dc)
        assert isclose(diffusivity_composite(Dd, Dc, fd=1), Dd)
        assert isclose(diffusivity_composite(Dd, Dc, fd=0.5),
                       (Dd+Dc)/2, rtol=0.1)
