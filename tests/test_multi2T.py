# noqa: N999
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pvlib import ivtools, pvsystem

import pvcircuit as pvc
from pvcircuit import Multi2T, Tandem3T

# Set to True once to write baseline test files, then revert to False
REGENERATE_TEST_FILES = False


@pytest.fixture
def dev2T():
    return Multi2T()


@pytest.fixture
def dev3T():
    return Tandem3T()


@pytest.fixture
def junction():
    return pvc.junction.Junction()


def test_2Tfrom3T(dev3T):

    dev2T = Multi2T.from_3T(dev3T)
    params2T = dev2T.MPP(pnts=150)
    _, params3T = dev3T.CM(pnts=150)

    np.testing.assert_almost_equal(params2T["Pmp"], params3T.Ptot)
    np.testing.assert_almost_equal(params2T["Imp"], params3T.Ito)
    np.testing.assert_almost_equal(params2T["Vmp"], -1 * params3T.Vtr, decimal=4)

    params3T = dev3T.Voc3()
    np.testing.assert_almost_equal(params2T["Voc"], -1 * params3T.Vtr)

    params3T = dev3T.Isc3()
    np.testing.assert_almost_equal(params2T["Isc"], params3T.Ito)


def test_2T_from_single_junction(junction):

    junction.set(n=[1], J0ratio=[1e4])

    dev2T = Multi2T.from_single_junction(junction)
    params2T = dev2T.MPP(pnts=150)
    # pvlib uses resistance_shunt; Gsh = 0 corresponds to Rsh = infinity (no shunt).
    resistance_shunt = np.inf if junction.Gsh == 0 else 1 / junction.Gsh
    pvlib_sd = pvsystem.singlediode(junction.Jext, junction.J0, junction.Rser, resistance_shunt, junction.n * junction.Vth)

    np.testing.assert_almost_equal(params2T["Pmp"], pvlib_sd.loc[0, "p_mp"], decimal=6)
    np.testing.assert_almost_equal(params2T["Imp"], pvlib_sd.loc[0, "i_mp"], decimal=5)
    np.testing.assert_almost_equal(params2T["Vmp"], pvlib_sd.loc[0, "v_mp"], decimal=4)
    np.testing.assert_almost_equal(params2T["Isc"], pvlib_sd.loc[0, "i_sc"])
    np.testing.assert_almost_equal(params2T["Voc"], pvlib_sd.loc[0, "v_oc"])


def test_multi2T_str(dev2T):

    test_file = "Multi2T_str.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(dev2T.__str__())

    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(test_str, dev2T.__str__())


def test_multi2T_setter(dev2T):
    # test setter of multi2T class

    dev2T.set(n=[1, 2])
    for junction in dev2T.j:
        np.testing.assert_array_equal(junction.n, np.array([1, 2]))

    dev2T.set(area=1.23)
    np.testing.assert_array_equal(dev2T.lightarea, 1.23)
    np.testing.assert_array_equal(dev2T.totalarea, 1.23)
    for junction in dev2T.j:
        np.testing.assert_array_equal(junction.lightarea, 1.23)
        np.testing.assert_array_equal(junction.totalarea, 1.23)

    with pytest.raises(ValueError, match=r"invalid class attribute test"):
        dev2T.set(test=-1)
    with pytest.raises(ValueError, match=r"invalid class attribute avalanche"):
        dev2T.set(avalanche=1)
    with pytest.raises(ValueError, match=r"invalid class attribute mrb"):
        dev2T.set(mrb=1)
    with pytest.raises(ValueError, match=r"invalid class attribute J0rb"):
        dev2T.set(J0rb=1)

    # dev2T.set(RBB="bishop")


def test_V2T(dev2T):
    # test 2T voltage from current
    np.testing.assert_almost_equal(dev2T.V2T(0), dev2T.Voc())
    # np.testing.assert_almost_equal(dev2T.V2T(dev2T.Isc()), 0)
    np.testing.assert_almost_equal(dev2T.V2T(0), dev2T.j[0].Vdiode(0) + dev2T.j[1].Vdiode(0))

    # np.testing.assert_almost_equal(dev2T.V2T(-1*dev2T.Isc()), 0) # TODO shouldn't voltage from current at Isc return 0?
    np.testing.assert_almost_equal(dev2T.V2T(-1 * dev2T.proplist("Jphoto")[0]), 0, decimal=5)

    np.testing.assert_almost_equal(dev2T.V2T(-1), np.nan)  # TODO consider brekdown here?
    # dev2T.set(RBB="bishop", Gsh=1e-4)


def test_Imaxrev(dev2T):
    # Series current is limited by the smallest junction capacity.

    np.testing.assert_almost_equal(dev2T.Imaxrev(), min(dev2T.j[0].Jext, dev2T.j[1].Jext))
    dev2T.j[0].set(Jext=1.2)
    np.testing.assert_almost_equal(dev2T.Imaxrev(), min(dev2T.j[0].Jext, dev2T.j[1].Jext))


def test_I2T(dev2T):
    # test 2T current from voltage
    # np.testing.assert_almost_equal(dev2T.I2T(0), -1 * dev2T.Imaxrev())
    np.testing.assert_almost_equal(dev2T.I2T(dev2T.Voc()), 0)
    np.testing.assert_almost_equal(dev2T.I2T(dev2T.V2T(0) * 1), 0)

    for i in np.arange(1e-6, dev2T.Voc()):
        np.testing.assert_almost_equal(dev2T.I2Troot(i), dev2T.I2T(i))


def test_MPP(dev2T):
    # calculate maximum power point and associated IV, Vmp, Imp, FF
    # res=0.001   #voltage resolution
    dev2T.set(Jext=0)
    np.testing.assert_equal(dev2T.MPP()["Pmp"], np.nan)


def test_4j():
    totalarea = 1.15
    tandem4J = pvc.Multi2T(name="4J", Eg_list=[1.83, 1.404, 1.049, 0.743], Jext=0.012, Rs2T=0.1, area=1)
    tandem4J.j[0].set(Jext=0.01196, n=[1, 1.6], J0ratio=[31, 4.5], totalarea=totalarea)
    tandem4J.j[1].set(Jext=0.01149, n=[1, 1.8], J0ratio=[17, 42], beta=14.3, totalarea=totalarea)
    tandem4J.j[2].set(Jext=0.01135, n=[1, 1.4], J0ratio=[51, 14], beta=8.6, totalarea=totalarea)
    tandem4J.j[3].set(Jext=0.01228, n=[1, 1.5], J0ratio=[173, 79], beta=10.5, totalarea=totalarea)
    tandem4J.j[3].RBB_dict = {"method": "JFG", "mrb": 43.0, "J0rb": 0.3, "Vrb": 0.0}

    mpp = tandem4J.MPP()

    # expected values updated for the tightened solver tolerance; the MPP grid
    # refinement lands on marginally different points (Pmp change < 1e-5 rel)
    np.testing.assert_allclose(mpp["Voc"], 3.4253298246871355, rtol=1e-5)
    np.testing.assert_allclose(mpp["Isc"], 0.011350990484439306, rtol=1e-5)
    np.testing.assert_allclose(mpp["Vmp"], 3.0124695153515013, rtol=1e-5)
    np.testing.assert_allclose(mpp["Imp"], 0.011074934395857742, rtol=1e-5)


def test_multi2T_copy(dev2T):
    """Multi2T.copy() must return an independent object: the junction
    list is duplicated and each junction inside it is itself a copy, so
    mutating any junction in the copy must NOT affect the original.
    Wrapper-level attribute Rs2T is independent too.
    """
    m2 = dev2T.copy()

    # wrapper objects are distinct
    assert m2 is not dev2T

    # junction list and elements are independent (not aliases)
    assert m2.j is not dev2T.j
    assert m2.j[0] is not dev2T.j[0]
    assert m2.j[1] is not dev2T.j[1]

    # Vmid array is independent too
    assert m2.Vmid is not dev2T.Vmid

    # initial contents match
    np.testing.assert_almost_equal(m2.Rs2T, dev2T.Rs2T)
    np.testing.assert_almost_equal(m2.j[0].Eg, dev2T.j[0].Eg)
    np.testing.assert_almost_equal(m2.j[1].Eg, dev2T.j[1].Eg)

    # wrapper-level attribute set via .set() is independent on the copy
    rs_before = dev2T.Rs2T
    m2.set(Rs2T=rs_before + 1.0)
    np.testing.assert_almost_equal(dev2T.Rs2T, rs_before)
    np.testing.assert_almost_equal(m2.Rs2T, rs_before + 1.0)

    # Mutating a junction on the copy must not affect the original
    eg_before = dev2T.j[0].Eg
    m2.j[0].set(Eg=eg_before + 0.1)
    np.testing.assert_almost_equal(dev2T.j[0].Eg, eg_before)
    np.testing.assert_almost_equal(m2.j[0].Eg, eg_before + 0.1)

    # MPP gives the same numerical result on both wrappers (initial state)
    m3 = dev2T.copy()
    mpp1 = dev2T.MPP()
    mpp2 = m3.MPP()
    np.testing.assert_allclose(mpp1["Voc"], mpp2["Voc"])
    np.testing.assert_allclose(mpp1["Isc"], mpp2["Isc"])


def test_multi2T_append_junction(dev2T, junction):
    """append_junction adds one Junction to the series stack, bumps
    njuncs and Vmid length, and updates Rs2T to a parallel-area sum."""
    n0 = dev2T.njuncs
    assert len(dev2T.Vmid) == n0

    dev2T.append_junction(junction)

    # one more junction in the stack
    assert dev2T.njuncs == n0 + 1
    assert len(dev2T.j) == n0 + 1
    assert len(dev2T.Vmid) == n0 + 1
    # newest junction is at the end and was copied (not the same object)
    assert dev2T.j[-1] is not junction
    np.testing.assert_array_equal(dev2T.j[-1].n, junction.n)
    np.testing.assert_almost_equal(dev2T.j[-1].Eg, junction.Eg)

    # Adding a second junction works identically
    dev2T.append_junction(pvc.junction.Junction())
    assert dev2T.njuncs == n0 + 2

    # MPP solver still produces a finite Voc with the extended stack
    mpp = dev2T.MPP()
    assert np.isfinite(mpp["Voc"])
    assert np.isfinite(mpp["Isc"])


def test_I2Troot_at_boundaries(dev2T):
    """I2Troot must agree with I2T at exactly V=Voc (current = 0) and
    return the (negative) short-circuit current at exactly V=0.  These
    boundary points are where the root finder is most likely to choke."""
    voc = dev2T.Voc()
    isc = dev2T.Isc()

    # At Voc both solvers must return ~0
    np.testing.assert_almost_equal(dev2T.I2T(voc), 0.0, decimal=5)
    np.testing.assert_almost_equal(dev2T.I2Troot(voc), 0.0, decimal=5)

    # At V=0 the bracketed solver must satisfy the requested terminal voltage,
    # even though the historical stepping solver stops at a nearby current.
    np.testing.assert_almost_equal(dev2T.I2T(0.0), -isc, decimal=5)
    isc_root = dev2T.I2Troot(0.0)
    assert abs(dev2T.V2T(isc_root)) < 1e-9
    assert isc_root < 0.0

    # Slightly beyond Voc: current must be strictly positive (forward injection)
    assert dev2T.I2T(voc + 1e-3) > 0
    assert dev2T.I2Troot(voc + 1e-3) > 0


def _mismatched_stacks():
    """Current-mismatched 2T stacks whose short circuit lies off the photocurrent boundaries."""

    def build(top=None, bot=None):
        dev3T = Tandem3T()
        dev3T.top.set(**(top or {}))
        dev3T.bot.set(**(bot or {}))
        return Multi2T.from_3T(dev3T)

    return {
        "top_limited": build(top=dict(Jext=0.012)),
        "bottom_limited_LC": build(bot=dict(Jext=0.012)),
        "bottom_limited_no_LC": build(bot=dict(Jext=0.012, beta=0.0)),
        "slightly_top_limited": build(top=dict(Jext=0.0139)),
        "top_limited_shunted": build(top=dict(Jext=0.012, Gsh=5e-4), bot=dict(Gsh=5e-4)),
        "top_limited_JFG": build(top=dict(Jext=0.012, RBB="JFG"), bot=dict(RBB="JFG")),
        "top_limited_bishop": build(top=dict(Jext=0.012, RBB="bishop", Gsh=1e-3), bot=dict(RBB="bishop", Gsh=1e-3)),
        "strongly_top_limited_leaky": build(top=dict(Jext=0.002, Gsh=2e-2), bot=dict(Gsh=2e-2)),
    }


@pytest.mark.parametrize("name", _mismatched_stacks())
def test_I2Troot_mismatched_stack(name):
    """I2Troot must find the current for any V <= Voc and agree with the stepping I2T.

    Regression: it raised 'Could not bracket I2T root' whenever the root was
    not at a junction photocurrent (shunt, breakdown, luminescent coupling)
    or lay on the saturated branch of an unshunted limiting junction.
    """
    dev2T = _mismatched_stacks()[name]
    for V in np.linspace(-0.5, dev2T.Voc(), 9):
        stepped = dev2T._I2T_stepping(V)  # resolves 1e-7 of the limiting current
        np.testing.assert_allclose(dev2T.I2Troot(V), stepped, rtol=1e-6, atol=1e-12)


def test_I2Troot_saturated_branch_returns_limiting_current():
    """Unshunted limiting junction: below its saturation the IV curve is vertical."""
    dev3T = Tandem3T()
    dev3T.top.set(Jext=0.012)
    dev2T = Multi2T.from_3T(dev3T)
    limiting = float(dev2T.j[0].Jext + dev2T.j[0].J0.sum())
    for V in (-1.0, 0.0, 0.3):
        np.testing.assert_allclose(dev2T.I2Troot(V), -limiting, rtol=1e-12)


class _Solver:
    """Context manager: run a block with pvcircuit.junction.SOLVER set to ``name``."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.previous = pvc.junction.SOLVER
        pvc.junction.SOLVER = self.name

    def __exit__(self, *exc):
        pvc.junction.SOLVER = self.previous


def _iv_stacks():
    """2T stacks for the I2T tests: matched, mismatched, shunted, with breakdown, 1 and 3 junctions."""

    def tandem(top=None, bot=None, Rs2T=0.0):
        dev = Multi2T()
        dev.j[0].set(**(top or {}))
        dev.j[1].set(**(bot or {}))
        dev.set(Rs2T=Rs2T)
        return dev

    return {
        "default": tandem(),
        "top_limited": tandem(top=dict(Jext=0.012)),
        "bottom_limited_LC": tandem(bot=dict(Jext=0.012)),
        "shunted_Rs2T": tandem(top=dict(Gsh=5e-4), bot=dict(Gsh=5e-4), Rs2T=2.0),
        "top_limited_JFG": tandem(top=dict(Jext=0.012, RBB="JFG"), bot=dict(RBB="JFG")),
        "three_junctions": Multi2T(Eg_list=[1.9, 1.4, 1.0]),
        "single_junction": Multi2T.from_single_junction(pvc.junction.Junction(Eg=1.12, Rser=0.5, Gsh=1e-4)),
    }


@pytest.mark.parametrize("name", _iv_stacks())
def test_I2T_array_matches_stepping(name):
    """All voltages solved together: the currents of the stepping algorithm, from reverse bias to beyond Voc."""
    dev = _iv_stacks()[name]
    Voc = dev.Voc()
    V = np.concatenate([np.linspace(-0.5, Voc, 14), Voc + np.array([0.02, 0.1])])
    current = dev.I2T(V)
    assert current.shape == V.shape and np.all(np.isfinite(current))
    assert np.all(np.diff(current) >= 0.0), "the current rises with the voltage"
    stepped = np.array([dev._I2T_stepping(v) for v in V])
    with _Solver("legacy"):
        np.testing.assert_array_equal(dev.I2T(V), stepped)  # the legacy setting is the stepping algorithm
    # the stepping stops within 1e-6 of the limiting current of the root
    np.testing.assert_allclose(current, stepped, rtol=0, atol=2e-6 * dev.Imaxrev())


def test_I2T_scalar_and_array_points_are_the_same_solve():
    dev = _iv_stacks()["bottom_limited_LC"]
    V = np.linspace(-0.3, dev.Voc() + 0.05, 7)
    current = dev.I2T(V)
    for voltage, expected in zip(V, current):
        scalar = dev.I2T(float(voltage))
        assert isinstance(scalar, float)
        assert scalar == pytest.approx(expected, rel=1e-11, abs=1e-16)
    assert dev.I2T(V.reshape(7, 1)).shape == (7, 1)
    assert dev.I2T(dev.Voc()) == 0.0


def test_I2T_roundtrip_where_the_curve_is_not_vertical():
    dev = _iv_stacks()["shunted_Rs2T"]
    V = np.linspace(-0.5, dev.Voc() + 0.1, 25)
    np.testing.assert_allclose(dev.V2T(dev.I2T(V)), V, rtol=0, atol=1e-8)


def test_I2T_saturated_branch_returns_limiting_current():
    """Unshunted limiting junction: below its saturation the IV curve is vertical."""
    dev = _iv_stacks()["top_limited"]
    limiting = float(dev.j[0].Jext + dev.j[0].J0.sum())
    np.testing.assert_allclose(dev.I2T(np.array([-1.0, 0.0, 0.3])), -limiting, rtol=1e-11)
    assert dev.Isc() == pytest.approx(limiting, rel=1e-11)


def test_I2T_voltage_out_of_range_is_nan():
    dev = Multi2T()  # no series resistance: two junctions cannot carry 50 V
    current = dev.I2T(np.array([1.0, 50.0]))
    assert np.isfinite(current[0]) and np.isnan(current[1])
    assert np.isnan(dev.I2T(50.0))
    # a voltage without solution must not leave the device in an undefined state
    assert np.all(np.isfinite(dev.Vmid)) and all(np.isfinite(junc.JLC) for junc in dev.j)


def test_I2T_leaves_junction_voltages_at_the_last_point():
    dev = _iv_stacks()["shunted_Rs2T"]
    V = np.array([0.4, 1.3])
    current = dev.I2T(V)
    np.testing.assert_allclose(dev.Vmid.sum() + dev.Rs2T * current[-1] / dev.totalarea, V[-1], rtol=0, atol=1e-8)


def test_V2T_many_points_match_few_points():
    """Both sides of SCALAR_SOLVE_CUTOFF give the same curve."""
    dev = _iv_stacks()["bottom_limited_LC"]
    current = np.linspace(-0.0119, 0.02, 150)
    many = dev.V2T(current)
    few = np.concatenate([dev.V2T(current[:75]), dev.V2T(current[75:])])
    np.testing.assert_allclose(many, few, rtol=0, atol=1e-11)
    with _Solver("legacy"):
        np.testing.assert_allclose(many, dev.V2T(current), rtol=0, atol=1e-10)


def test_V2T_in_breakdown_with_an_overflowing_derivative():
    """JFG breakdown with a small mrb: the derivative overflows inside the bracket of the junction solve.

    Regression: a zero Newton step from the infinite derivative counted as
    converged, and V2T was off by up to 8.56 V at I = -0.0125 A.
    """
    dev = Multi2T(Eg_list=[1.8, 1.4])
    for junc in dev.j:
        junc.set(RBB="JFG", Gsh=1e-4)
        junc.set(mrb=0.5)
    dev.j[0].set(Jext=0.010)
    dev.j[1].set(Jext=0.014)
    current = np.linspace(-0.0125, -0.0095, 11)
    fast = dev.V2T(current)
    with _Solver("legacy"):
        legacy = dev.V2T(current)
    np.testing.assert_allclose(fast, legacy, rtol=0, atol=1e-9)


def test_I2T_negative_photocurrent_on_an_unshunted_junction():
    """Without a shunt a junction with Jph < 0 carries no current below -totalarea * Jph.

    Regression: V2T had no solution between 0 and that current, the forward
    bracket closed on 0, and every voltage gave nan.
    """
    dev = Multi2T(Eg_list=[1.8, 1.4])
    dev.j[0].set(Jext=-0.0005)
    dev.j[1].set(Jext=0.014)
    V = np.array([2.0, 2.4])
    current = dev.I2T(V)
    assert np.all(np.isfinite(current))
    np.testing.assert_allclose(dev.V2T(current), V, rtol=0, atol=1e-9)
    # roots by bisection on the finite branch, current > 0.0005 A
    np.testing.assert_allclose(current, [0.000500950033841535, 0.0029084425271042706], rtol=1e-9)
    assert dev.I2T(2.0) == current[0]
    # below about 1.17 V the curve is vertical at the floor 0.0005 A (within totalarea * sum(J0) below
    # it, or 1e-13 A above it): the current is the floor, as on the saturated reverse branch
    vertical = dev.I2T(np.array([-9.0, 1.0, 1.1]))
    np.testing.assert_allclose(vertical, 0.0005, rtol=1e-9)
    np.testing.assert_allclose(vertical[1], 0.0004999999999962281, rtol=1e-9)


def test_I2T_negative_photocurrent_on_a_shunted_junction():
    """With a shunt the junction with Jph < 0 carries current below -totalarea * Jph: the root can lie between 0 and that floor.

    Regression: the forward bracket started at the floor without checking that
    the root lies above it, and these points were nan.
    """
    dev = Multi2T(Eg_list=[1.8, 1.4])
    dev.j[0].set(Jext=-0.0005, Gsh=1e-3)
    dev.j[1].set(Jext=0.014, Gsh=1e-3)
    current = dev.I2T(0.8)
    np.testing.assert_allclose(current, 0.00026458049688785046, rtol=1e-9)  # root by bisection
    assert dev.V2T(current) == pytest.approx(0.8, abs=1e-9)
    # Voc < 0: the short-circuit current is a forward current below the floor of 0.005 A
    rows = dev.MPP_rows([25.0], [[1.8, 1.4]], 0.0, [[-0.005, 0.014]])
    np.testing.assert_allclose(rows["Isc"], 0.003955882274302504, rtol=1e-9)
    dev.j[0].set(Jext=-0.005)
    np.testing.assert_allclose(dev.MPP()["Isc"], 0.003955882274302504, rtol=1e-9)


def test_MPP_negative_photocurrent_on_an_unshunted_junction():
    """Isc is the floor of the vertical branch, but no point of the scan [-Isc, 0] has a finite power.

    Regression: np.argmax of the all-nan power is 0, and MPP reported Imp = Isc.
    """
    dev = Multi2T()
    dev.j[0].set(Jext=-0.002)
    dev.j[1].set(Jext=0.025433)
    mpp = dev.MPP()
    assert mpp["Isc"] == abs(dev.I2T(0.0))  # the floor, 0.002 A
    for key in ("Vmp", "Imp", "Pmp", "FF"):
        assert np.isnan(mpp[key]), key
    np.testing.assert_array_equal(dev.Ipoints, [-mpp["Isc"], np.nan, 0.0])


@pytest.mark.parametrize("name", _iv_stacks())
def test_MPP_matches_legacy_solver(name):
    dev = _iv_stacks()[name]
    fast = dev.MPP()
    with _Solver("legacy"):
        legacy = dev.MPP()
        assert legacy["Isc"] == abs(dev._I2T_stepping(0.0))
    for key in ("Voc", "Isc", "Vmp", "Imp", "Pmp", "FF"):
        assert fast[key] == pytest.approx(legacy[key], rel=1e-6), key


def plot_2T():

    dev2T = Multi2T()
    # dev2T.set(RBB="bishop")
    # dev2T.j[0].set(Vrb=-2)
    # dev2T.j[1].set(Vrb=-2)
    volts = np.linspace(-1, dev2T.Voc(), 500)
    currs1 = []
    currs2 = []

    t_start = time.perf_counter()
    for v in volts:
        currs1.append(dev2T.I2T(v))
    t_end = time.perf_counter()
    print(f"Timef or Dans {t_end-t_start}s")

    t_start = time.perf_counter()
    for v in volts:
        currs2.append(dev2T.I2Troot(v))
    t_end = time.perf_counter()
    print(f"Timef or Dans {t_end-t_start}s")

    fig, ax = plt.subplots()
    ax.plot(volts, currs1, ".", ms=1)
    ax.plot(volts, currs2, "o", ms=5, mfc="None")
    plt.show()


def i2trun():
    dev2T = Multi2T()

    dev2T.I2T(2.4)
    dev2T.I2Troot(2.4)


def generate_test_files():
    """Generate all baseline test files. Run: python tests/test_multi2T.py"""
    global REGENERATE_TEST_FILES
    REGENERATE_TEST_FILES = True

    dev2T = Multi2T()
    print("Generating Multi2T_str.txt...")
    test_multi2T_str(dev2T)

    REGENERATE_TEST_FILES = False
    print("Done!")


if __name__ == "__main__":
    generate_test_files()
