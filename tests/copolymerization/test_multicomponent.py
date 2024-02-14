# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
import pytest
from numpy import all, isclose

from polykin.copolymerization import (TerminalModel, convert_Qe_to_r,
                                      inst_copolymer_binary,
                                      inst_copolymer_multi,
                                      inst_copolymer_ternary,
                                      monomer_drift_multi, sequence_multi,
                                      transitions_multi, tuples_multi)
from polykin.utils.exceptions import ShapeError


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
    F_m = inst_copolymer_multi(np.array([f1, f2, 1 - f1 - f2]), r)
    assert all(isclose(F_t, F_m,))  # type: ignore
    with pytest.raises(ValueError):
        _ = inst_copolymer_multi([0.1, 0.5, 0.4], r, P=r)
    with pytest.raises(ValueError):
        _ = inst_copolymer_multi([0.1, 0.5, 0.4], None, P=r)


def test_monomer_drift_multi():
    r1 = 0.4
    r2 = 0.7
    r = np.ones((2, 2))
    r[0, 1] = r1
    r[1, 0] = r2
    x = [0.1, 0.4, 0.8, 0.99]
    m = TerminalModel(r1, r2)
    for f10 in [0.2, 0.5, 0.9]:
        sol_b = m.drift(f10, x)
        sol_m = monomer_drift_multi([f10, 1. - f10], r, x)
        assert all(isclose(sol_b, sol_m[:, 0], rtol=1e-4))
    with pytest.raises(ShapeError):
        _ = monomer_drift_multi([f10, 1. - f10], np.ones((3, 3)), x)


def test_convert_Qe():
    Qe1 = (1., -0.8)     # Sty
    Qe2 = (0.78, 0.4)    # MMA
    Qe3 = (0.026, -0.88)  # VAc
    r = convert_Qe_to_r([Qe1, Qe2, Qe3])
    assert all(isclose(r.diagonal(), [1., 1., 1.]))
    assert all(isclose(r[0, :], [1., 0.5, 40.], rtol=0.1))


def test_transitions_multi():
    r = np.ones((3, 3))
    r[0, 1] = 0.2
    r[1, 0] = 2.3
    r[0, 2] = 3.0
    r[2, 0] = 0.9
    r[1, 2] = 0.4
    r[2, 1] = 1.5
    f = [0.5, 0.3, 0.2]
    P = transitions_multi(f, r)
    assert all(isclose(P.diagonal(), [0.241, 0.295, 0.209], rtol=1e-2))
    assert all(isclose(P.sum(axis=-1), np.ones(3)))


def test_transitions_multi_2():
    r1 = 0.4
    r2 = 0.7
    r = np.ones((2, 2))
    r[0, 1] = r1
    r[1, 0] = r2
    m = TerminalModel(r1, r2)
    for f1 in [0.2, 0.5, 0.9]:
        sol_b = m.transitions(f1)
        sol_m = transitions_multi([f1, 1. - f1], r)
        for i in range(2):
            for j in range(2):
                assert (isclose(sol_b[f"{i+1}{j+1}"], sol_m[i, j]))


def test_sequence_multi():
    r1 = 0.4
    r2 = 0.7
    r = np.ones((2, 2))
    r[0, 1] = r1
    r[1, 0] = r2
    m = TerminalModel(r1, r2)
    for f1 in [0.2, 0.5, 0.9]:
        f = [f1, 1. - f1]
        k = [1, 3, 5]
        P = transitions_multi(f, r)
        # Savg
        sol_b = list(m.sequence(f1).values())
        sol_m = sequence_multi(P.diagonal())
        assert np.all(isclose(sol_b, sol_m))  # type: ignore
        # SLD
        sol_b = m.sequence(f1, k)
        sol_m = sequence_multi(P.diagonal(), k)
        assert np.all(isclose(sol_b['1'], sol_m[0]))


def test_tuples_multi():
    r1 = 0.4
    r2 = 0.7
    r = np.ones((2, 2))
    r[0, 1] = r1
    r[1, 0] = r2
    m = TerminalModel(r1, r2)
    for f1 in [0.2, 0.5, 0.9]:
        f = [f1, 1. - f1]
        P = transitions_multi(f, r)
        sol_b = m.triads(f1)
        sol_m = tuples_multi(P, 3)
        for idx in sol_m.keys():
            idx_str = f"{idx[0]+1}{idx[1]+1}{idx[2]+1}"
            try:
                sol = sol_b[idx_str]
            except KeyError:
                idx_str = idx_str[::-1]
                sol = sol_b[idx_str]
            assert (
                isclose(sol_m[idx], sol))
