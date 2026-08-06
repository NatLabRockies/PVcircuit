# noqa: N999
import itertools
import re
from pathlib import Path

import numpy as np
import pytest

import pvcircuit as pvc
from pvcircuit import IV3T, Multi2T, Tandem3T

# Set to True once to write baseline test files, then revert to False
REGENERATE_TEST_FILES = False


@pytest.fixture
def dev2T():
    return Multi2T()


@pytest.fixture
def dev3T():
    return Tandem3T()


@pytest.fixture
def iv3t():
    return IV3T()


@pytest.fixture
def junction():
    return pvc.junction.Junction()


def test_tandem3T_str(dev3T):

    test_file = "Tandem3T_str.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(dev3T.__str__())

    # read fixed test case for s-type
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", test_str))


def test_tandem3T_maxsetters(dev3T):
    # test the area and TC setter

    top_area = dev3T.top.totalarea
    bot_area = dev3T.bot.totalarea

    tc = max(dev3T.top.TC, dev3T.bot.TC)

    np.testing.assert_almost_equal(dev3T.totalarea, max(top_area, bot_area))
    np.testing.assert_almost_equal(dev3T.TC, tc)

    set_area = max(top_area, bot_area) * 100
    dev3T.top.set(area=set_area)
    np.testing.assert_almost_equal(dev3T.totalarea, set_area)
    np.testing.assert_almost_equal(dev3T.lightarea, set_area)

    set_TC = tc * 200
    dev3T.top.set(TC=set_TC)
    np.testing.assert_almost_equal(dev3T.TC, set_TC)


def test_set(dev3T):
    dev3T.set(n=[2,3], TC=20)
    np.testing.assert_almost_equal(dev3T.TC, 20)
    np.testing.assert_almost_equal(dev3T.top.TC, 20)
    np.testing.assert_almost_equal(dev3T.bot.TC, 20)
    np.testing.assert_array_almost_equal(dev3T.top.n, [2,3])
    np.testing.assert_array_almost_equal(dev3T.bot.n, [2,3])


def test_V3T(dev3T, iv3t):
    # test 3T voltage from current
    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)
    iv3t.convert("I", "load2dev")
    dev3T.V3T(iv3t)

    test_file = "Tandem3T_V3T-s.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    # read fixed test case for s-type
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))

    dev3T.bot.set(Jext=20e-3, pn=-1)
    dev3T.V3T(iv3t)

    test_file = "Tandem3T_V3T-r.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    # read fixed test case for r-type
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))

    # also test top beta
    dev3T.top.set(beta=0.1)
    dev3T.V3T(iv3t)
    iv3t.__str__()
    np.testing.assert_almost_equal(np.nanmax(iv3t.Izo), 0.055)


def test_J3Tabs(dev3T, iv3t):
    # test 3T voltage from current
    iv3t.box("VA", -1.5, 0.2, 30, "VB", -1.5, 0.2, 30)
    iv3t.convert("V", "load2dev")
    dev3T.J3Tabs(iv3t)

    test_file = "Tandem3T_J3TAabs-s.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    # read fixed test case for s-type
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))

    dev3T.bot.set(pn=-1)
    dev3T.J3Tabs(iv3t)

    test_file = "Tandem3T_J3TAabs-r.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    # read fixed test case for r-type
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))

    # also test top beta
    dev3T.top.set(beta=0.2)
    dev3T.J3Tabs(iv3t)
    iv3t.__str__()
    np.testing.assert_almost_equal(iv3t.Ito.max(), 8109.48136792229)


def test_I3Trel(dev3T, iv3t):
    # test 3T voltage from current
    iv3t.box("VA", -1.5, 0.2, 30, "VB", -1.5, 0.2, 30)
    iv3t.convert("V", "load2dev")
    dev3T.I3Trel(iv3t)

    test_file = "Tandem3T_I3Trel-s.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    # read fixed test case for s-type
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))

    dev3T.bot.set(pn=-1)
    dev3T.I3Trel(iv3t)

    test_file = "Tandem3T_I3Trel-r.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    # read fixed test case for r-type
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))

    # also test top beta
    dev3T.top.set(beta=0.2)
    dev3T.I3Trel(iv3t)
    finite = np.isfinite(iv3t.Iro) & np.isfinite(iv3t.Izo) & np.isfinite(iv3t.Ito)
    assert np.count_nonzero(finite) > 0.95 * finite.size
    assert np.nanmax(np.abs(iv3t.Iro + iv3t.Izo + iv3t.Ito)) < 1e-10


def test_scaled_newton_system_handles_mixed_units():
    jacobian = np.array(
        [
            [[1e-20, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3e20]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    expected = np.array([1.0, -2.0, 3.0])
    residual = np.vstack((-jacobian[0] @ expected, np.ones(3)))

    delta = pvc.tandem3T._solve_scaled_newton_system(jacobian, residual)

    np.testing.assert_allclose(delta[0], expected, rtol=1e-14)
    assert np.all(np.isnan(delta[1]))


def test_I3Trel_vectorized_matches_brent_fallback(dev3T):
    vectorized = IV3T(name="vectorized", meastype="CZ")
    vectorized.box("VA", -1.2, 0.1, 9, "VB", -1.2, 0.1, 9)
    vectorized.convert("V", "load2dev")
    fallback = vectorized.copy()

    dev3T.I3Trel(vectorized)
    dev3T.copy()._i3t_brent_fallback(fallback)

    for key in vectorized.Idevlist:
        np.testing.assert_allclose(getattr(vectorized, key), getattr(fallback, key), rtol=1e-9, atol=1e-10, equal_nan=True)


def test_VM(dev3T):

    ratios = [(1, 1), (2, 1), (3, 2)]
    for ratio in ratios:
        iv3t_vm, iv3t_vmpp = dev3T.VM(*ratio)

        vm_fname = "Tandem3T_iv3t_vm_{}.txt".format("".join(map(str, ratio)))
        vmpp_fname = "Tandem3T_iv3t_vmpp_{}.txt".format("".join(map(str, ratio)))
        if REGENERATE_TEST_FILES:
            with open(pvc.pvcpath.parent.joinpath("tests", "test_files", vm_fname), "w", encoding="utf8") as fout:
                fout.write(iv3t_vm.__str__())
            with open(pvc.pvcpath.parent.joinpath("tests", "test_files", vmpp_fname), "w", encoding="utf8") as fout:
                fout.write(iv3t_vmpp.__str__())

        # read fixed test case for s-type
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", vm_fname), "r", encoding="utf8") as fin:
            test_vm = fin.read()
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", vmpp_fname), "r", encoding="utf8") as fin:
            test_vmpp = fin.read()

        np.testing.assert_string_equal(re.sub(r"\s+", " ", test_vm), re.sub(r"\s+", " ", iv3t_vm.__str__()))
        np.testing.assert_string_equal(re.sub(r"\s+", " ", test_vmpp), re.sub(r"\s+", " ", iv3t_vmpp.__str__()))


def test_CM(dev3T):
    dev2T = Multi2T.from_3T(dev3T)
    lnout, mpp = dev3T.CM()
    v2t = lnout.VA - lnout.VB
    i2t = []
    for v in v2t:
        i2t.append(dev2T.I2T(v))
    np.testing.assert_array_almost_equal(i2t, lnout.IA)

    mpp2t = dev2T.MPP()
    np.testing.assert_almost_equal(mpp2t["Pmp"], mpp.Ptot[0], decimal=6)
    np.testing.assert_almost_equal(mpp2t["Vmp"], mpp.VA - mpp.VB, decimal=3)
    np.testing.assert_almost_equal(mpp2t["Imp"], -mpp.IA, decimal=5)
    np.testing.assert_almost_equal(mpp2t["Imp"], mpp.IB, decimal=5)
    np.testing.assert_almost_equal(mpp2t["Imp"], -mpp.Iro, decimal=5)
    np.testing.assert_almost_equal(mpp2t["Imp"], mpp.Ito, decimal=5)


def test_MPP(dev3T):
    # Test the 4T MPP operating point.
    mpp = dev3T.MPP()
    mpp = dev3T.MPP(bplot=True)

    # Use single junctions for comparison
    tc = Multi2T.from_single_junction(dev3T.top)
    bc = Multi2T.from_single_junction(dev3T.bot)

    tc_mpp = tc.MPP(pnts=30)
    bc_mpp = bc.MPP(pnts=30)

    mpp = dev3T.MPP(pnts=30)

    np.testing.assert_almost_equal(tc_mpp["Pmp"] + bc_mpp["Pmp"], mpp.Ptot[0], decimal=4)
    np.testing.assert_almost_equal(tc_mpp["Imp"], mpp.Ito[0], decimal=4)
    np.testing.assert_almost_equal(bc_mpp["Imp"], -mpp.Iro[0], decimal=4)


def test_VI0(dev3T):

    constraints = {
        "VztIro": ("Vzt", "Iro"),
        "VrzIto": ("Vrz", "Ito"),
        "VtrIzo": ("Vtr", "Izo"),
    }
    for point, (voltage_key, current_key) in constraints.items():

        iv3t = dev3T.VI0(point)

        assert abs(getattr(iv3t, voltage_key)[0]) < 1e-6
        assert abs(getattr(iv3t, current_key)[0]) < 1e-12
        assert abs(iv3t.Iro[0] + iv3t.Izo[0] + iv3t.Ito[0]) < 1e-12
        assert abs(iv3t.Vzt[0] + iv3t.Vrz[0] + iv3t.Vtr[0]) < 1e-12
        if point == "VtrIzo":
            assert abs(iv3t.Vtr[0]) < 1e-9

        test_file = f"Tandem3T_VI0_{point}.txt"
        if REGENERATE_TEST_FILES:
            with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
                fout.write(iv3t.__str__())

        # read fixed test case for s-type
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
            test_str = fin.read()

        np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str).strip(), re.sub(r"\s+", " ", iv3t.__str__()).strip())


def test_VIpoints(dev3T):

    iv3t = IV3T()

    current_keys = [k for k in iv3t.arraykeys if k.startswith("I") and len(k) > 2]
    voltage_keys = [k for k in iv3t.arraykeys if k.startswith("V") and len(k) > 2]

    iv3t = dev3T.VIpoint("Iro", "Izo", "Vzt")
    iv3t = dev3T.VIpoint("Iro", "Izo", "Vzt", bplot=True)
    # test a few randomly
    combs = list(itertools.product(range(len(current_keys)), range(len(current_keys)), range(len(voltage_keys))))
    combs = [combo for combo in combs if combo[0] != combo[1]]
    combids = [4, 42, 21, 43, 23]
    for combid in combids:
        combo = combs[combid]
        iv3t = dev3T.VIpoint(current_keys[combo[0]], current_keys[combo[1]], voltage_keys[combo[2]])

        test_file = f"Tandem3T_VIpoint_{iv3t.name}.txt"
        if REGENERATE_TEST_FILES:
            with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
                fout.write(iv3t.__str__())

        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
            test_str = fin.read()

        np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))

    #     iv3t_vals = np.concatenate([getattr(iv3t,k) for k in iv3t.arraykeys])
    #     if all(~np.isnan(iv3t_vals)):
    #         testids.append(combid)

    # print(testids)
    # import random
    # random_ids = random.sample(testids, 5)


def test_specialpoints(dev3T):

    special_points = dev3T.specialpoints()

    test_file = "Tandem3T_specialpoints.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(special_points.__str__())

    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", special_points.__str__()))


def test_tandem3T_copy(dev3T):
    """Tandem3T.copy() returns an independent device: top and bot are
    duplicated via Junction.copy() so mutating either on the copy must
    NOT propagate to the original.  Wrapper-level Rz is independent.
    """
    d2 = dev3T.copy()

    assert d2 is not dev3T
    # top/bot junctions are independent
    assert d2.top is not dev3T.top
    assert d2.bot is not dev3T.bot
    np.testing.assert_almost_equal(d2.top.Eg, dev3T.top.Eg)
    np.testing.assert_almost_equal(d2.bot.Eg, dev3T.bot.Eg)

    rz_before = dev3T.Rz
    d2.set(Rz=rz_before + 0.5)
    np.testing.assert_almost_equal(dev3T.Rz, rz_before)
    np.testing.assert_almost_equal(d2.Rz, rz_before + 0.5)

    # Mutating top junction on the copy does NOT affect the original
    eg_top_before = dev3T.top.Eg
    d2.top.set(Eg=eg_top_before + 0.1)
    np.testing.assert_almost_equal(dev3T.top.Eg, eg_top_before)
    np.testing.assert_almost_equal(d2.top.Eg, eg_top_before + 0.1)

    # Voc3 still works on a freshly-copied device and returns the same triple
    d3 = dev3T.copy()
    np.testing.assert_almost_equal(dev3T.Voc3().Vzt[0], d3.Voc3().Vzt[0])


def test_Rz_sensitivity(dev3T):
    """Rz is the resistance between bottom and middle contact, so it
    only matters when Izo != 0.  At a fixed device operating point with
    non-zero Izo, increasing Rz must change Vrz by I*Rz (Ohm's law on
    the Rz path).  Voc3 (all currents zero) is independent of Rz.
    """
    voc3_lowRz = dev3T.Voc3().Vzt[0]

    # Pick a 1-point IV3T with non-zero Izo and let V3T solve voltages
    iv_low = IV3T(name="probe", shape=1, meastype="CZ", area=dev3T.lightarea)
    iv_low.set(Iro=-5e-3, Izo=5e-3, Ito=0.0)
    dev3T.V3T(iv_low)

    rz_before = dev3T.Rz
    dev3T.set(Rz=rz_before + 50.0)

    iv_high = IV3T(name="probe", shape=1, meastype="CZ", area=dev3T.lightarea)
    iv_high.set(Iro=-5e-3, Izo=5e-3, Ito=0.0)
    dev3T.V3T(iv_high)

    voc3_highRz = dev3T.Voc3().Vzt[0]

    # Voc3 (zero-current) is independent of Rz
    np.testing.assert_almost_equal(voc3_lowRz, voc3_highRz, decimal=6)

    # At Izo=5 mA (per cm^2), Vrz shifts by (Rz_new - Rz_old) * Izo
    # Iro/Izo/Ito are stored in mA in IV3T after V3T, but Rz acts on A;
    # the relevant invariant is that Vrz CHANGED in the expected direction
    # by a non-trivial amount (> 1 mV).
    delta_vrz = abs(iv_high.Vrz[0] - iv_low.Vrz[0])
    assert delta_vrz > 1e-3


def test_LC_effect_on_voc3(dev3T):
    """At Voc3 (all device currents zero) the top junction sits at
    V_top > 0, so Jem(V_top) > 0 and the LC current JLC = beta * Jem
    pumps additional photocurrent into the bottom junction.  To keep
    Izo=0 the bottom junction must supply more recombination current,
    which requires a larger |V_bot|.  V_bot enters Voc3 as Vrz, so
    |Vrz| increases when beta is turned on.

    Conversely Isc3 (Vzt=Vrz=Vtr=0) is LC-insensitive because V_top=0
    forces the EL term in Jem to zero, and the PL term (Lan and Green
    2015) is also zero here because the fixture uses the default
    gamma=0.  Setting gamma>0 would give a nonzero PL baseline at
    short circuit even with V_top=0.
    """
    voc_base = dev3T.Voc3()
    isc_base = dev3T.Isc3()

    dev3T.top.set(beta=0.5)
    voc_lc = dev3T.Voc3()
    isc_lc = dev3T.Isc3()

    # |Vrz| (bottom-junction voltage at Izo=0) grows under LC
    assert abs(voc_lc.Vrz[0]) > abs(voc_base.Vrz[0])
    # Isc3 is unchanged by LC by construction (V_top=0 -> Jem=0)
    np.testing.assert_almost_equal(isc_lc.Iro[0], isc_base.Iro[0], decimal=8)
    np.testing.assert_almost_equal(isc_lc.Ito[0], isc_base.Ito[0], decimal=8)


def test_Multi2T_from_3T_preserves_junction_params(dev3T):
    """Multi2T.from_3T copies junction parameters by default.  The
    resulting 2T device must have matching Eg, n, J0ratio on both
    junctions."""
    dev2T = Multi2T.from_3T(dev3T, copy_attributes=True)

    assert dev2T.njuncs == 2
    np.testing.assert_almost_equal(dev2T.j[0].Eg, dev3T.top.Eg)
    np.testing.assert_almost_equal(dev2T.j[1].Eg, dev3T.bot.Eg)
    np.testing.assert_array_almost_equal(dev2T.j[0].n, dev3T.top.n)
    np.testing.assert_array_almost_equal(dev2T.j[0].J0ratio, dev3T.top.J0ratio)
    np.testing.assert_array_almost_equal(dev2T.j[1].n, dev3T.bot.n)
    np.testing.assert_array_almost_equal(dev2T.j[1].J0ratio, dev3T.bot.J0ratio)

    # copy_attributes=True yields disconnected junctions (different objects)
    assert dev2T.j[0] is not dev3T.top
    assert dev2T.j[1] is not dev3T.bot

    # copy_attributes=False yields dynamically-connected junctions
    dev2T_dyn = Multi2T.from_3T(dev3T, copy_attributes=False)
    assert dev2T_dyn.j[0] is dev3T.top
    assert dev2T_dyn.j[1] is dev3T.bot


def generate_test_files():
    """Generate all baseline test files. Run: python tests/test_tandem3T.py"""
    global REGENERATE_TEST_FILES
    REGENERATE_TEST_FILES = True

    dev3T = Tandem3T()
    iv3t = IV3T()

    print("Generating Tandem3T_str.txt...")
    test_tandem3T_str(dev3T)
    dev3T = Tandem3T()
    iv3t = IV3T()
    print("Generating Tandem3T_V3T*.txt...")
    test_V3T(dev3T, iv3t)
    dev3T = Tandem3T()
    iv3t = IV3T()
    print("Generating Tandem3T_J3TAabs*.txt...")
    test_J3Tabs(dev3T, iv3t)
    dev3T = Tandem3T()
    iv3t = IV3T()
    print("Generating Tandem3T_I3Trel*.txt...")
    test_I3Trel(dev3T, iv3t)
    dev3T = Tandem3T()
    print("Generating Tandem3T_VM*.txt...")
    test_VM(dev3T)
    dev3T = Tandem3T()
    print("Generating Tandem3T_VI0*.txt...")
    test_VI0(dev3T)
    dev3T = Tandem3T()
    print("Generating Tandem3T_VIpoint*.txt...")
    test_VIpoints(dev3T)
    dev3T = Tandem3T()
    print("Generating Tandem3T_specialpoints.txt...")
    test_specialpoints(dev3T)

    REGENERATE_TEST_FILES = False
    print("Done!")


if __name__ == "__main__":
    generate_test_files()


    # current_keys = [k for k in iv3t.arraykeys if k.startswith("I") and len(k) > 2]
    # voltage_keys = [k for k in iv3t.arraykeys if k.startswith("V") and len(k) > 2]

    # # test a few randomly
    # combs = list(itertools.product(range(len(current_keys)), range(len(current_keys)), range(len(voltage_keys))))
    # combs = [combo for combo in combs if combo[0] != combo[1]]
    # combids = range(len(combs))
    # testids = []
    # testnames = []
    # for combid in combids:
    #     combo = combs[combid]
    #     iv3t = dev3T.VIpoint(current_keys[combo[0]], current_keys[combo[1]], voltage_keys[combo[2]])
    #     iv3t_vals = np.concatenate([getattr(iv3t, k) for k in iv3t.arraykeys])
    #     if all(~np.isnan(iv3t_vals)):
    #         if iv3t.name not in testnames:
    #             testids.append(combid)
    #             testnames.append(iv3t.name)

    # print(testnames)
    # print(testids)
    # import random

    # random_ids = random.sample(testids, 5)
    # print(random_ids)
    # print(testnames)