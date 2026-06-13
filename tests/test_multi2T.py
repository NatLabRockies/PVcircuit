import os
import time
from pathlib import Path

import matplotlib.pyplot as plt

# Set to True once to write baseline test files, then revert to False
REGENERATE_TEST_FILES = False
import numpy as np
import pytest
from pvlib import ivtools, pvsystem

import pvcircuit as pvc
from pvcircuit import Multi2T, Tandem3T


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
    # Maximum rev bias current?
    # TODO: check behaviour and use reverse bias

    np.testing.assert_almost_equal(dev2T.Imaxrev(), max(dev2T.j[0].Jext, dev2T.j[1].Jext))
    dev2T.j[0].set(Jext=1.2)
    np.testing.assert_almost_equal(dev2T.Imaxrev(), max(dev2T.j[0].Jext, dev2T.j[1].Jext))


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

    np.testing.assert_allclose(mpp["Voc"], 3.425330977574876, rtol=1e-5)
    np.testing.assert_allclose(mpp["Voc"], 3.425330977574876, rtol=1e-5)
    np.testing.assert_allclose(mpp["Isc"], 0.011350991117526023, rtol=1e-5)
    np.testing.assert_allclose(mpp["Vmp"], 3.0129358068534247, rtol=1e-5)
    np.testing.assert_allclose(mpp["Imp"], 0.011073118854968986, rtol=1e-5)


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

    # At V=0 both solvers must return -Isc (current extraction convention)
    np.testing.assert_almost_equal(dev2T.I2T(0.0), -isc, decimal=5)
    np.testing.assert_almost_equal(dev2T.I2Troot(0.0), -isc, decimal=5)

    # Slightly beyond Voc: current must be strictly positive (forward injection)
    assert dev2T.I2T(voc + 1e-3) > 0
    assert dev2T.I2Troot(voc + 1e-3) > 0


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
    print("Generating Multi2T_str.txt..."); test_multi2T_str(dev2T)

    REGENERATE_TEST_FILES = False
    print("Done!")


if __name__ == "__main__":
    generate_test_files()
