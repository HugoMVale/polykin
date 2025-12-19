# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import inf, isclose

from polykin.hmt.hex import U_cylindrical_wall, U_plane_wall


def test_U_plane_wall():
    assert isclose(U_plane_wall(h := 1e2, inf, 0, 1), h)
    assert isclose(U_plane_wall(h := 1e2, h, 0, 1), h / 2)
    assert isclose(U_plane_wall(inf, inf, 1, k := 3), k)


def test_U_cylindrical_wall():
    Uo = U_cylindrical_wall(hi := 1e3, inf, di := 1e-2, do := 2e-2, inf)
    assert isclose(Uo, hi * di / do)
    Uo = U_cylindrical_wall(inf, ho := 2e3, di := 1e-2, do := 2e-2, inf)
    assert isclose(Uo, ho)
    Uo = U_cylindrical_wall(hi := 1e3, hi, di := 1, di + 1e-10, inf)
    assert isclose(Uo, hi / 2, rtol=1e-6)
    Uo_plane = U_plane_wall(inf, inf, L := 1e-2, k := 50)
    Uo_cylinder = U_cylindrical_wall(inf, inf, di := 1e3, di + 2 * L, k)
    assert isclose(Uo_plane, Uo_cylinder, rtol=1e-5)
