"""Smoke tests for the area-handling convention: area-normalised quantities
(series resistance, power density, efficiency) refer to the device area
A_ref = device.totalarea = max_i(j_i.totalarea).

The pvcircuit code is exercised today with equal-area junctions, but
these tests are future-proofing against mismatched-area devices (e.g.
3T back-contact tandems where the top emitter is smaller than the
bottom absorber). Each test below would silently pass before the
area-handling refactor, so the goal is to lock in the new behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

import pvcircuit as pvc
from pvcircuit import IV3T, Multi2T, Tandem3T
from pvcircuit.junction import Junction

# ---------------------------------------------------------------------------
# A3: Junction.set must reject lightarea > totalarea
# ---------------------------------------------------------------------------


def test_junction_set_rejects_lightarea_gt_totalarea():
    """lightarea cannot exceed totalarea (Jphoto would otherwise inflate)."""
    j = Junction()
    # Equal-area path still works.
    j.set(lightarea=0.5, totalarea=1.0)
    # Mismatched-area path is rejected.
    with pytest.raises(ValueError, match="lightarea"):
        j.set(lightarea=2.0, totalarea=1.0)
    # Direct totalarea shrink below an existing lightarea is also rejected.
    j.set(area=2.0)  # both = 2
    with pytest.raises(ValueError, match="lightarea"):
        j.set(totalarea=0.5)


# ---------------------------------------------------------------------------
# A1: Multi2T.from_3T composes Rs2T consistently with V2T(I)
# ---------------------------------------------------------------------------


def _build_mismatched_3T(rser_top=1.0, rser_bot=1.0, a_top=1.0, a_bot=2.0):
    """Helper: Tandem3T with arbitrary top/bot areas and Rser values."""
    dev = Tandem3T()
    dev.top.set(Rser=rser_top, area=a_top)
    dev.bot.set(Rser=rser_bot, area=a_bot)
    return dev


def test_from_3T_Rs2T_matches_series_resistance_formula():
    """Rs2T should equal sum_i(Rser_i * A_ref / A_i), A_ref = max(A_i).

    For Rser_top = Rser_bot = 1 Ohm*cm^2, A_top = 1, A_bot = 2:
        R_abs = 1/1 + 1/2 = 1.5 Ohm
        Rs2T  = R_abs * A_ref = 1.5 * 2 = 3.0 Ohm*cm^2
    """
    dev3T = _build_mismatched_3T(rser_top=1.0, rser_bot=1.0, a_top=1.0, a_bot=2.0)
    dev2T = Multi2T.from_3T(dev3T)
    np.testing.assert_allclose(dev2T.Rs2T, 3.0, rtol=1e-12)
    # Equal-area collapse: Rs2T = sum(Rser_i)
    dev3T_eq = _build_mismatched_3T(rser_top=1.5, rser_bot=2.5, a_top=1.0, a_bot=1.0)
    dev2T_eq = Multi2T.from_3T(dev3T_eq)
    np.testing.assert_allclose(dev2T_eq.Rs2T, 4.0, rtol=1e-12)


def test_from_3T_Rs2T_voltage_drop_is_self_consistent():
    """The Rs2T * I / totalarea drop that V2T applies must equal the
    analytical absolute drop R_abs * I (with R_abs in Ohm)."""
    dev3T = _build_mismatched_3T(rser_top=0.5, rser_bot=2.0, a_top=1.0, a_bot=2.0)
    dev2T = Multi2T.from_3T(dev3T)
    I_test = 0.012  # 12 mA, arbitrary
    R_abs = 0.5 / 1.0 + 2.0 / 2.0  # = 1.5 Ohm
    drop_expected = R_abs * I_test
    drop_actual = dev2T.Rs2T * I_test / dev2T.totalarea
    np.testing.assert_allclose(drop_actual, drop_expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# A2: Multi2T.append_junction composes Rs2T consistently
# ---------------------------------------------------------------------------


def test_append_junction_Rs2T_matches_series_resistance_formula():
    """Starting from a single-junction Multi2T (created with from_single_junction
    so Rs2T equals that junction's Rser), appending a second junction must
    update Rs2T to sum_i(Rser_i * A_ref / A_i)."""
    j_top = Junction(Eg=1.8, Rser=0.5, area=1.0)
    j_bot = Junction(Eg=1.4, Rser=2.0, area=2.0)

    dev = Multi2T.from_single_junction(j_top)
    # Single junction: Rs2T = Rser_top * A_ref / A_top = 0.5 * 1 / 1 = 0.5
    np.testing.assert_allclose(dev.Rs2T, 0.5, rtol=1e-12)

    dev.append_junction(j_bot)
    # Two junctions: R_abs = 0.5/1 + 2.0/2 = 1.5; A_ref = 2 -> Rs2T = 3.0
    np.testing.assert_allclose(dev.Rs2T, 3.0, rtol=1e-12)

    drop_expected = 1.5 * 0.012
    drop_actual = dev.Rs2T * 0.012 / dev.totalarea
    np.testing.assert_allclose(drop_actual, drop_expected, rtol=1e-12)


def test_append_junction_matches_from_3T_for_equivalent_stack():
    """from_3T and (from_single_junction + append_junction) on the same
    physical stack must yield identical Rs2T -- the two code paths must
    agree."""
    dev3T = _build_mismatched_3T(rser_top=0.3, rser_bot=1.7, a_top=1.0, a_bot=3.0)
    dev2T_from = Multi2T.from_3T(dev3T)

    dev2T_app = Multi2T.from_single_junction(dev3T.top.copy())
    dev2T_app.append_junction(dev3T.bot.copy())

    np.testing.assert_allclose(dev2T_from.Rs2T, dev2T_app.Rs2T, rtol=1e-12)


# ---------------------------------------------------------------------------
# B5/B6: Pmp_density and efficiency() consistency
# ---------------------------------------------------------------------------


def test_Multi2T_MPP_has_Pmp_density_consistent_with_Pmp():
    """MPP['Pmp_density'] * totalarea must equal MPP['Pmp']."""
    dev = Multi2T()  # default 1 cm^2
    mpp = dev.MPP()
    assert "Pmp_density" in mpp
    np.testing.assert_allclose(mpp["Pmp_density"] * dev.totalarea, mpp["Pmp"], rtol=1e-12)


def test_Multi2T_efficiency_matches_manual_calculation():
    """device.efficiency('global') = Pmp / (Pin * totalarea) with
    Pin from qe.PintMD('global') in W/cm^2."""
    from pvcircuit import qe

    dev = Multi2T()
    eff = dev.efficiency("global")
    Pin = float(qe.PintMD("global")) / 1e4  # W/m^2 -> W/cm^2
    Pmp = dev.MPP()["Pmp"]
    expected = Pmp / (Pin * dev.totalarea)
    np.testing.assert_allclose(eff, expected, rtol=1e-10)
    # 2J III-V tandem under AM1.5G should be a sensible efficiency.
    assert 0.0 < eff < 0.6


def test_Tandem3T_efficiency_matches_manual_calculation():
    from pvcircuit import qe

    dev = Tandem3T()
    eff = dev.efficiency("global")
    Pin = float(qe.PintMD("global")) / 1e4
    pt = dev._best_mpp_point()
    expected = float(pt.Ptot[0]) / (Pin * dev.totalarea)
    np.testing.assert_allclose(eff, expected, rtol=1e-10)
    # 3T two-load efficiency must be >= the series-connected 2T efficiency of the same device
    assert eff >= Multi2T.from_3T(dev).efficiency("global")
    assert 0.0 < eff < 0.6


# ---------------------------------------------------------------------------
# C8: Tandem3T.J3Tabs honours the IV3T absolute-Amps contract
# ---------------------------------------------------------------------------


def test_J3Tabs_writes_absolute_currents_with_mismatched_areas():
    """J3Tabs writes ABSOLUTE currents [A] (= Jdensity * junction_area),
    not per-junction densities, into iv3T.(Iro, Izo, Ito). Doubling
    bot.totalarea at the same diode voltages must double Iro because
    Iro = Jr * bot.totalarea (Jr unchanged for the same Vrmid).

    Rz is set to 0 here so Vt and Vr are decoupled from the device
    totalarea (Jz is then determined by Kirchhoff in J3Tabs).
    """
    voltages = dict(Vzt=0.4, Vrz=-0.3, Vtr=-0.1)

    dev_eq = _build_mismatched_3T(rser_top=0.1, rser_bot=0.1, a_top=1.0, a_bot=1.0)
    dev_eq.set(Rz=0.0)
    iv_eq = IV3T(name="eq", shape=1, meastype="CZ", area=dev_eq.totalarea)
    iv_eq.set(**voltages)
    assert dev_eq.J3Tabs(iv_eq) == 0

    dev_mis = _build_mismatched_3T(rser_top=0.1, rser_bot=0.1, a_top=1.0, a_bot=2.0)
    dev_mis.set(Rz=0.0)
    iv_mis = IV3T(name="mis", shape=1, meastype="CZ", area=dev_mis.totalarea)
    iv_mis.set(**voltages)
    assert dev_mis.J3Tabs(iv_mis) == 0

    # Top junction unchanged -> same absolute Ito
    np.testing.assert_allclose(float(iv_mis.Ito[0]), float(iv_eq.Ito[0]), rtol=1e-9)
    # Bot junction has 2x area -> Iro doubles (absolute Amps)
    np.testing.assert_allclose(
        float(iv_mis.Iro[0]), 2.0 * float(iv_eq.Iro[0]), rtol=1e-9
    )
