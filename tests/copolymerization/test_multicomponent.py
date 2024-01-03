# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from numpy import all, isclose

from polykin.copolymerization import (convert_Qe_to_r, inst_copolymer_binary,
                                      inst_copolymer_multicomponent,
                                      inst_copolymer_ternary)


def test_inst_copolymer_ternary():
    r12 = 0.2
    r21 = 2.3
    r13 = 3.0
    r31 = 0.9
    r23 = 0.4
    r32 = 1.5
    # f3=0
    f1 = 0.4
    f2 = 1 - f1
    F1_b = inst_copolymer_binary(f1, r1=r12, r2=r21)
    F1_t, _, _ = inst_copolymer_ternary(f1, f2, r12, r21, r13, r31, r23, r32)
    assert isclose(F1_b, F1_t)
    # f2=0
    f1 = 0.4
    f2 = 0.
    F1_b = inst_copolymer_binary(f1, r1=r13, r2=r31)
    F1_t, _, _ = inst_copolymer_ternary(f1, f2, r12, r21, r13, r31, r23, r32)
    # f1=0
    f1 = 0.
    f2 = 0.5
    F2_b = inst_copolymer_binary(f2, r1=r23, r2=r32)
    _, F2_t, _ = inst_copolymer_ternary(f1, f2, r12, r21, r13, r31, r23, r32)
    assert isclose(F2_b, F2_t)


def test_inst_copolymer_multicomponent():
    r12 = 0.2
    r21 = 2.3
    r13 = 3.0
    r31 = 0.9
    r23 = 0.4
    r32 = 1.5
    f1 = 0.2
    f2 = 0.5
    F_t = inst_copolymer_ternary(f1, f2, r12, r21, r13, r31, r23, r32)
    r = np.ones((3, 3))
    r[0, 1] = r12
    r[1, 0] = r21
    r[0, 2] = r13
    r[2, 0] = r31
    r[1, 2] = r23
    r[2, 1] = r32
    F_m = inst_copolymer_multicomponent(np.array([f1, f2]), r)
    assert all(isclose(F_t, F_m,))  # type: ignore


def test_convert_Qe():
    Qe1 = (1., -0.8)     # Sty
    Qe2 = (0.78, 0.4)    # MMA
    Qe3 = (0.026, -0.88)  # VAc
    r = convert_Qe_to_r([Qe1, Qe2, Qe3])
    assert all(isclose(r.diagonal(), [1., 1., 1.]))
    assert all(isclose(r[0, :], [1., 0.5, 40.], rtol=0.1))
