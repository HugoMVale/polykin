# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import pytest
from numpy import inf, isclose

from polykin.transport.hmt import (Nu_combined, Nu_cylinder, Nu_cylinder_bank,
                                   Nu_cylinder_free, Nu_drop, Nu_plate,
                                   Nu_plate_free, Nu_sphere, Nu_sphere_free,
                                   Nu_tank, Nu_tube, U_cylindrical_wall,
                                   U_plane_wall)


def test_Nu_tube():
    # turbulent
    Pr = 1
    Re = 1e5
    Nu = Nu_tube(Re, Pr)
    Nu_Colburn = 0.023*Re**0.8 * Pr**0.33
    assert isclose(Nu, Nu_Colburn, rtol=10e-2)
    # laminar
    Re = 1e3
    D_L = 0.1
    Nu_Walas = (3.66**3 + 1.61**3 * Re * Pr * D_L)**(1/3)
    Nu = Nu_tube(Re, Pr, D_L)
    assert isclose(Nu, Nu_Walas, rtol=10e-2)


def test_Nu_cylinder_cross_flow():
    "Incropera, Example 7.4,  p. 373"
    Pr = 0.7
    Re = 6071
    Nu = Nu_cylinder(Re, Pr)
    assert isclose(Nu, 40.6, rtol=1e-2)


def test_Nu_sphere():
    Pr = 0.709
    Re = 6510
    mur = 181.6/197.8
    Nu = Nu_sphere(Re, Pr, mur)
    assert isclose(Nu, 47.4, rtol=1e-2)


def test_Nu_drop():
    Pr = 1.
    Re = 1000.
    Nu_s = Nu_sphere(Re, Pr, 1.0)
    Nu_d = Nu_drop(Re, Pr)
    assert isclose(Nu_s, Nu_d, rtol=5e-2)


def test_Nu_plate():
    "Incropera, Example 7.1,  p. 360"
    Pr = 0.687
    Re = 9597
    Nu = Nu_plate(Re, Pr)
    assert isclose(Nu, 57.4, rtol=1e-2)
    Re = 6.84e5
    Nu = Nu_plate(Re, Pr)
    assert isclose(Nu, 753, rtol=1e-2)


def test_Nu_tank():
    "Handbook of Industrial Mixing, p. 882"
    # Example 14.1
    Re = 1.57e6
    Pr = 4.37
    Nu = Nu_tank('wall', '6BD', Re, Pr, mur=1.)
    assert isclose(Nu, 16400, rtol=1e-2)
    # Example 14.2
    Re = 76
    Pr = 2.5e5
    Nu = Nu_tank('wall', 'helical-ribbon', Re, Pr, mur=1.)
    assert isclose(Nu, 337, rtol=1e-2)


def test_Nu_tank_2():
    "Simple test, to check all branches."
    surface_impeller_map = {
        'wall': ['6BD', '4BF', '4BP', 'HE3', 'PROP', 'anchor', 'helical-ribbon'],
        'bottom-head': ['6BD', '4BF', '4BP', 'HE3'],
        'helical-coil': ['PROP', '6BD'],
        'harp-coil-0': ['4BF'],
        'harp-coil-45': ['6BD']
    }
    for surface, impellers in surface_impeller_map.items():
        for impeller in impellers:
            for Re in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
                Pr = 1
                Nu = Nu_tank(surface, impeller, Re, Pr, mur=1)  # type: ignore
                assert Nu >= 0
    with pytest.raises(ValueError):
        _ = Nu_tank('X', 'HE3', Re, Pr, mur=1)  # type: ignore
    with pytest.raises(ValueError):
        _ = Nu_tank('wall', 'X', Re, Pr, mur=1)  # type: ignore
    with pytest.raises(ValueError):
        _ = Nu_tank('bottom-head', 'anchor', Re, Pr, mur=1)
    with pytest.raises(ValueError):
        _ = Nu_tank('helical-coil', 'HE3', Re, Pr, mur=1)
    with pytest.raises(ValueError):
        _ = Nu_tank('harp-coil-0', 'HE3', Re, Pr, mur=1)
    with pytest.raises(ValueError):
        _ = Nu_tank('harp-coil-45', 'HE3', Re, Pr, mur=1)


def test_Nu_cylinder_bank():
    "Handbook of Industrial Mixing, p. 383"
    # Example 7.6
    Nu = Nu_cylinder_bank(v=6, rho=1, mu=14.82e-6, Pr=0.710, Prs=0.701,
                          aligned=False, D=16.4e-3, ST=31.3e-3, SL=34.3e-3, NL=7)
    assert isclose(Nu, 87.9, rtol=2e-2)


def test_Nu_cylinder_free():
    Ra = 5.073e6
    Pr = 0.697
    Nu = Nu_cylinder_free(Ra, Pr)
    assert isclose(Nu, 23.3, rtol=1e-2)


def test_Nu_sphere_free():
    Ra = 5e6
    Pr = 0.7
    Nu = Nu_sphere_free(Ra, Pr)
    assert isclose(Nu, 23.5, rtol=0.01)


def test_Nu_plate_free():
    # Vertical: Incropera, Example 9.2,  p. 395
    Ra = 1.813e9
    Pr = 0.69
    Nu = Nu_plate_free('vertical', Ra, Pr)
    assert isclose(Nu, 147, rtol=0.01)
    with pytest.raises(ValueError):
        _ = Nu_plate_free('vertical', Ra)
    # Horizontal: Incropera, Example 9.3,  p. 499
    Ra = 1.38e8
    Nu_top = Nu_plate_free('horizontal-upper-heated', Ra)
    Nu_bottom = Nu_plate_free('horizontal-lower-heated', Ra)
    assert isclose(Nu_top*0.0265/0.375, 5.47, rtol=0.01)
    assert isclose(Nu_bottom*0.0265/0.375, 2.07, rtol=0.01)
    # Horizontal-continuity
    assert isclose(Nu_plate_free('horizontal-upper-heated', 1e7-1),
                   Nu_plate_free('horizontal-upper-heated', 1e7+1),
                   rtol=0.10)
    # Overall
    with pytest.raises(ValueError):
        _ = Nu_plate_free('X', 1e8, 1)


def test_U_plane_wall():
    assert isclose(U_plane_wall(h := 1e2, inf, 0, 1), h)
    assert isclose(U_plane_wall(h := 1e2, h, 0, 1), h/2)
    assert isclose(U_plane_wall(inf, inf, 1, k := 3), k)


def test_U_cylindrical_wall():
    Uo = U_cylindrical_wall(hi := 1e3, inf, di := 1e-2, do := 2e-2, inf)
    assert isclose(Uo, hi*di/do)
    Uo = U_cylindrical_wall(inf, ho := 2e3, di := 1e-2, do := 2e-2, inf)
    assert isclose(Uo, ho)
    Uo = U_cylindrical_wall(hi := 1e3, hi, di := 1, di + 1e-10, inf)
    assert isclose(Uo, hi/2, rtol=1e-6)
    Uo_plane = U_plane_wall(inf, inf, L := 1e-2, k := 50)
    Uo_cylinder = U_cylindrical_wall(inf, inf, di := 1e3, di + 2*L, k)
    assert isclose(Uo_plane, Uo_cylinder, rtol=1e-5)


def test_Nu_combined():
    assert isclose(Nu_combined(20, 10, True), 20.80, rtol=1e-2)
    assert isclose(Nu_combined(20, 10, False), 19.13, rtol=1e-2)
