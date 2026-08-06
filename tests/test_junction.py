import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pvlib import ivtools, pvsystem

import pvcircuit as pvc
from pvcircuit import Multi2T

# Set to True once to write baseline test files, then revert to False
REGENERATE_TEST_FILES = False


# %%
def test_basic_functions():

    # Test thermal voltage
    np.testing.assert_almost_equal(pvc.junction.Vth(25), 0.02569257912108585)

    # Test temperature conversion
    np.testing.assert_almost_equal(pvc.junction.TK(25), 298.15)

    # Test detailed balance current
    Eg = 1.12
    TC = 25
    EgkT = Eg / pvc.junction.Vth(TC)
    np.testing.assert_almost_equal(pvc.junction.Jdb(TC=TC, Eg=Eg, sigma=0), 6.249646867228706e-17)

    # Comapre to old PV with sigma = 0
    np.testing.assert_almost_equal(pvc.junction.Jdb(TC=TC, Eg=Eg, sigma=0), pvc.junction.DB_PREFIX * pvc.junction.TK(TC) ** 3.0 * (EgkT * EgkT + 2.0 * EgkT + 2.0) * np.exp(-EgkT))


# %%
@pytest.fixture
def junction_2d():
    return pvc.junction.Junction()


# %%
@pytest.fixture
def junction_1d():
    junc = pvc.junction.Junction(
        name="Junction 1Diode",
        Eg=1.25,
        sigma=2e-4,
        TC=25,
        Gsh=2e-5,
        Rser=2.33e-3,
        area=1.0,
        n=[1.32],  # pvc needs n as list
        # J0ref=[2.3e-14],  # A/cm^2 pvc needs J0ref as list
        Jext=0.04,  # A/cm^2
        beta=0.0,
    )
    return junc


def test_junction_str(junction_2d):

    test_file = "Junction.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(junction_2d.__str__())

    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(test_str, junction_2d.__str__())


def test_junction_setter(junction_2d):
    """
    Test the junction setters.
    """

    # test setting of n
    junction_2d.set(n=[1, 2])
    np.testing.assert_array_equal(junction_2d.n, np.array([1, 2]))

    # test setting of single n value
    junction_2d.set(**{"n[0]": 3})
    np.testing.assert_array_equal(junction_2d.n, np.array([3, 2]))

    # test mismatch when setting single n value
    with pytest.raises(IndexError, match=r"invalid junction index. Set index is 3 but junction size is 2"):
        junction_2d.set(**{"n[2]": 4})

    # test mismatch when setting n and J0ratio of different size
    with pytest.raises(ValueError, match=r"n and J0ratio must be same size"):
        junction_2d.set(n=[1, 2, 3], J0ratio=[1, 2])

    with pytest.raises(ValueError, match=r"n and J0ratio must be same size"):
        junction_2d.set(n=[1, 2], J0ratio=[1, 2, 3])

    # test mismatch when setting n or J0ratio with different number of current diode number
    with pytest.raises(ValueError, match=r"setting single n or J0ratio value must match previous number of diodes"):
        junction_2d.set(n=[1, 2, 3])

    with pytest.raises(ValueError, match=r"setting single n or J0ratio value must match previous number of diodes"):
        junction_2d.set(J0ratio=[1, 2, 3])

    # test setting the general area with light and total area
    junction_2d.set(area=1.23)
    np.testing.assert_almost_equal(junction_2d.lightarea, 1.23)
    np.testing.assert_almost_equal(junction_2d.totalarea, 1.23)

    # test setting invalid class values
    with pytest.raises(ValueError, match=r"invalid class attribute test"):
        junction_2d.set(test=-1)
    with pytest.raises(ValueError, match=r"invalid class attribute avalanche"):
        junction_2d.set(avalanche=1)
    with pytest.raises(ValueError, match=r"invalid class attribute mrb"):
        junction_2d.set(mrb=1)

    # test reverse bias breakdown model keys
    junction_2d.set(RBB="bishop")
    junction_2d.set(avalanche=1)

    with pytest.raises(ValueError, match=r"invalid class attribute J0rb"):
        junction_2d.set(J0rb=1)

    junction_2d.set(RBB="JFG")
    junction_2d.set(J0rb=1)


def test_junction_properties(junction_2d):
    """
    Test the junction properties.
    """

    np.testing.assert_almost_equal(junction_2d.Jphoto, 0.04)
    np.testing.assert_allclose(junction_2d.J0, [1.3141250302231388e-15, 3.6250862475576206e-09])


def test_junction_j0init(junction_2d):

    with pytest.raises(ValueError, match=r"J0ref and n must be same size"):
        junction_2d._J0init(1e-15)

    junction_2d._J0init([1e-15, 1e-9])

    np.testing.assert_allclose(junction_2d.J0, [1e-15, 1e-9])
    np.testing.assert_allclose(junction_2d.J0ratio, [7.609626, 2.75855506])


def test_j0_log_domain_avoids_intermediate_underflow():
    junction = pvc.junction.Junction(n=[0.039], J0ratio=[1e300])

    with np.errstate(under="ignore"):
        direct_power = (junction.Jdb / pvc.junction.J0_REFERENCE) ** (1.0 / junction.n[0])
    assert direct_power == 0.0

    expected_log_j0 = (
        np.log(pvc.junction.J0_REFERENCE)
        + np.log(junction.J0ratio[0])
        + np.log(junction.Jdb / pvc.junction.J0_REFERENCE) / junction.n[0]
    )
    saturation_current = junction.J0
    assert np.log(saturation_current[0]) == pytest.approx(expected_log_j0)

    junction._J0init(saturation_current)
    np.testing.assert_allclose(junction.J0ratio, [1e300], rtol=1e-12)


def test_recombination_product_avoids_intermediate_overflow():
    junction = pvc.junction.Junction(n=[0.1], J0ref=[1e-300], Jext=0.0)
    state = junction._solver_state()
    voltage = 720.0 * junction.n[0] * junction.Vth
    expected = np.exp(np.log(junction.J0[0]) + 720.0)

    with np.errstate(over="ignore"):
        assert np.isinf(junction.J0[0] * np.expm1(720.0))

    vectorized = pvc.junction._recomb_current(np.array([voltage]), state)[0]
    scalar = pvc.junction._recomb_current_scalar(voltage, state)
    np.testing.assert_allclose(vectorized, expected, rtol=1e-13)
    np.testing.assert_allclose(scalar, expected, rtol=1e-13)


def test_scaled_derivative_recovers_underflowed_exponential():
    expected = np.exp(np.log(1e300) - 800.0)

    with np.errstate(under="ignore"):
        assert 1e300 * np.exp(-800.0) == 0.0

    actual = pvc.junction._scaled_exp(1e300, -800.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-13)


def test_jem(junction_2d):

    np.testing.assert_almost_equal(junction_2d.Jem(0.6), 1.8227873411146403e-06)
    np.testing.assert_almost_equal(junction_2d.Jem(-0.6), 0.0)


def test_jem_pl_at_zero_bias(junction_2d):
    """PL coupling (Lan and Green 2015, Eq. 2b) is voltage-independent
    and must remain present at short circuit and reverse bias.  With
    'gamma > 0' and 'Jphoto > 0' the PL contribution dominates at
    'Vmid <= 0' and the EL contribution is absent there.  This matches
    the nonzero V_top=0 LC baseline reported by Tayagaki et al. 2018
    (Fig. 5b).
    """
    junction_2d.set(gamma=0.1)
    expected_pl = 0.1 * junction_2d.Jphoto

    # At and below short circuit only the PL term contributes.
    np.testing.assert_almost_equal(junction_2d.Jem(0.0), expected_pl)
    np.testing.assert_almost_equal(junction_2d.Jem(-0.6), expected_pl)

    # Above short circuit the EL term adds on top of the PL baseline.
    el_term = junction_2d.Jdb * (np.exp(0.6 / junction_2d.Vth) - 1.0)
    np.testing.assert_almost_equal(junction_2d.Jem(0.6), el_term + expected_pl)


def test_notdiode(junction_2d):

    assert not junction_2d.notdiode()
    junction_2d.set(J0ratio=[0, 0])
    assert junction_2d.notdiode()


def test_Jmultidiodes(junction_2d):

    np.testing.assert_almost_equal(junction_2d.Jmultidiodes(0.56), 0.00019985765193700707)


def test_JshuntRBB(junction_2d):

    junction_2d.set(RBB=None, Gsh=1e-4)
    np.testing.assert_almost_equal(junction_2d.JshuntRBB(2), 0.0002)
    junction_2d.set(RBB="JFG")
    np.testing.assert_almost_equal(junction_2d.JshuntRBB(-3), -3.033350517003164)
    junction_2d.set(RBB="bishop")
    np.testing.assert_almost_equal(junction_2d.JshuntRBB(-18), -0.0018153659543416658)


def test_Vdiode(junction_2d):

    # expected values updated for the tightened solver tolerance (xtol 1e-4 -> 1e-11);
    # shifts are < 1e-5 V, within the old solver's own error bars
    np.testing.assert_almost_equal(junction_2d.Vdiode(0), 0.7849544537913467)
    np.testing.assert_almost_equal(junction_2d.Vdiode(-40e-3), 0, decimal=5)
    np.testing.assert_equal(junction_2d.Vdiode(-41e-3), np.nan)  # VLIM_REVERSE

    junction_2d.set(RBB="bishop", Gsh=1e-4)
    np.testing.assert_almost_equal(junction_2d.Vdiode(-41e-3), -9.652390520409787)
    junction_2d.set(RBB="JFG")
    np.testing.assert_almost_equal(junction_2d.Vdiode(-41e-3), -0.9224612808482097)

    junction_2d.set(J0ratio=[0, 0])
    np.testing.assert_almost_equal(junction_2d.Vdiode(-41e-3), 0)


def test_Vmid(junction_2d):

    np.testing.assert_almost_equal(junction_2d.Vmid(0), 0)
    np.testing.assert_almost_equal(junction_2d.Vmid(0.5), 0.5)

    junction_2d.set(Rser=0.73)
    np.testing.assert_almost_equal(junction_2d.Vmid(0), 0.029200002624152344)
    np.testing.assert_almost_equal(junction_2d.Vmid(0.5), 0.5291205858003947)


def test_JV(junction_1d):
    # Compare measurement, pvlib single-diode fit, and pvcircuit Junction forward model.
    # pvlib single-diode forward model -- feed Rs in V/(mA/cm^2) to match the
    # mA/cm^2 current space that i_from_v expects (undo the \Omega*cm^2 conversion).
    # photocurrent, saturation_current, Rs_fit_ohm, Rsh_fit_ohm, nNsVth = jv_fits[name]
    # J_pvlib = ivtools.i_from_v(V_meas, photocurrent, saturation_current, Rs_fit_ohm, Rsh_fit_ohm, nNsVth)

    # pvcircuit Junction model for subcells
    voc = junction_1d.Vdiode(0)
    V_sweep = np.linspace(0, voc, 200)
    V_mid = np.vectorize(junction_1d.Vmid)(V_sweep)
    J_pvc_sweep = np.vectorize(junction_1d.Jparallel)(V_mid, junction_1d.Jphoto) * 1  # A/cm^2
    J_pvlib = pvsystem.i_from_v(V_sweep, junction_1d.Jphoto, junction_1d.J0, junction_1d.Rser, 1/junction_1d.Gsh, junction_1d.n[0] * pvc.junction.Vth(junction_1d.TC))
    np.testing.assert_allclose(J_pvc_sweep, J_pvlib, rtol=1e-3, atol=1e-6)
    # fig,ax = plt.subplots()
    # ax.plot(V_sweep, J_pvc_sweep, label="pvcircuit Junction")
    # ax.plot(V_sweep, J_pvlib, label="pvlib single-diode fit")
    # plt.show()


def test_junction_copy(junction_2d):
    """Junction.copy() must produce an independent object: mutating one
    must not affect the other.  n, J0ratio, and RBB_dict are duplicated
    explicitly (deepcopy crashes per the class docstring)."""

    junction_2d.set(RBB="JFG")
    j_copy = junction_2d.copy()

    # different python objects, same content
    assert j_copy is not junction_2d
    np.testing.assert_array_equal(j_copy.n, junction_2d.n)
    np.testing.assert_array_equal(j_copy.J0ratio, junction_2d.J0ratio)
    assert j_copy.RBB_dict == junction_2d.RBB_dict
    assert j_copy.Eg == junction_2d.Eg
    assert j_copy.TC == junction_2d.TC

    # the three explicitly-copied containers must be independent
    assert j_copy.n is not junction_2d.n
    assert j_copy.J0ratio is not junction_2d.J0ratio
    assert j_copy.RBB_dict is not junction_2d.RBB_dict

    # mutate the copy through the controlled setter; original stays put
    n_before = junction_2d.n.copy()
    j_copy.set(n=[1.5, 2.5])
    np.testing.assert_array_equal(junction_2d.n, n_before)
    np.testing.assert_array_equal(j_copy.n, [1.5, 2.5])

    j_copy.set(RBB=None)
    assert junction_2d.RBB_dict["method"] == "JFG"
    assert j_copy.RBB_dict["method"] is None


def test_junction_Jparallel(junction_2d):
    """Jparallel(V, Jtot) zeroes out at the V that Vdiode(Jdiode) finds."""

    # At Vdiode(Jdiode=0), Jtot = Jphoto and Jparallel must vanish
    # to the scalar Brent voltage tolerance.
    v = junction_2d.Vdiode(0)
    residual = junction_2d.Jparallel(v, junction_2d.Jphoto)
    np.testing.assert_almost_equal(residual, 0.0, decimal=5)

    # For a notdiode (sum(J0)=0), Jparallel returns Jtot unchanged.
    junction_2d.set(J0ratio=[0, 0])
    np.testing.assert_almost_equal(junction_2d.Jparallel(0.5, 1.234), 1.234)


def test_junction_RBB_dict(junction_2d):
    """RBB shortcut populates a method-specific parameter dict; switching
    methods rebuilds the dict; bishop vs JFG give different breakdown shapes."""

    # 'JFG' shortcut
    junction_2d.set(RBB="JFG")
    assert junction_2d.RBB_dict["method"] == "JFG"
    for key in ("mrb", "J0rb", "Vrb"):
        assert key in junction_2d.RBB_dict

    # 'bishop' shortcut rebuilds dict with different keys (no J0rb, has avalanche)
    junction_2d.set(RBB="bishop")
    assert junction_2d.RBB_dict["method"] == "bishop"
    assert "avalanche" in junction_2d.RBB_dict
    assert "J0rb" not in junction_2d.RBB_dict

    # Setting J0rb while in bishop mode is rejected
    with pytest.raises(ValueError, match=r"invalid class attribute J0rb"):
        junction_2d.set(J0rb=1)

    # Any other RBB value disables breakdown (method=None)
    junction_2d.set(RBB="not_a_real_method")
    assert junction_2d.RBB_dict == {"method": None}
    # And JshuntRBB then only returns the ohmic shunt Vd * Gsh
    junction_2d.set(Gsh=1e-4)
    np.testing.assert_almost_equal(junction_2d.JshuntRBB(-2.0), -2.0 * 1e-4)


def generate_test_files():
    """Generate all baseline test files. Run: python tests/test_junction.py"""
    global REGENERATE_TEST_FILES
    REGENERATE_TEST_FILES = True

    junction = pvc.junction.Junction()
    print("Generating Junction.txt...")
    test_junction_str(junction)

    REGENERATE_TEST_FILES = False
    print("Done!")


if __name__ == "__main__":
    pytest.main(["-v", __file__])

    generate_test_files()

    # root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # fp = os.path.join(root, "data", "Pvsk_1.70MA-free_JV.csv")
    # #     fp = os.path.join(root,"IBC2x2.csv")

    # A = 0.122
    # TC = 25  # [degC]
    # Eg = 1.8  # [eV]

    # data = pd.read_csv(fp)
    # # Measured terminal voltage.
    # voltage = data["v"].to_numpy(np.double)  # [V]
    # # Measured terminal current.
    # current = data["i"].to_numpy(np.double) / 1000 * A  # [A]

    # sort_id = np.argsort(voltage)

    # voltage = voltage[sort_id]
    # current = current[sort_id]

    # photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = ivtools.sde.fit_sandia_simple(voltage, current)
    # d_fitres = pvsystem.singlediode(photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth, ivcurve_pnts=100, method="brentq")

    # fit_voltage = d_fitres["v"]
    # fit_current = d_fitres["i"]

    # Jext = photocurrent / A  # [A/cm^2]
    # n = nNsVth / pvc.junction.Vth(TC)
    # J0ref = saturation_current / A
    # Rser = resistance_series * A
    # Gsh = 1 / (resistance_shunt * A)
    # # pvc.junction.DB_PREFIX
    # Jdb = pvc.junction.Jdb(TC=TC, Eg=Eg)
    # J0ratio = (J0ref / pvc.junction.J0_REFERENCE) / (Jdb / pvc.junction.J0_REFERENCE) ** (1.0 / n)

    # PVK = Multi2T(name="Psk", area=A, Jext=Jext, Eg_list=[Eg], n=[n], J0ratio=[J0ratio])
    # PVK.set(Rs2T=Rser, Gsh=Gsh)
    # PVK.j[0]

    # MPP = PVK.MPP()

    # Voc = MPP["Voc"]
    # Isc = MPP["Isc"]

    # pvc_voltage_set = np.linspace(0, Voc)
    # pvc_current_set = np.linspace(0, Isc)

    # pvc_voltage_calc = np.zeros_like(pvc_voltage_set)
    # pvc_current_calc = np.zeros_like(pvc_current_set)

    # V2Tvect = np.vectorize(PVK.V2T)
    # I2Tvect = np.vectorize(PVK.I2T)

    # pvc_current_calc = I2Tvect(pvc_voltage_set)
    # pvc_voltage_calc = V2Tvect(pvc_current_set)

    # Vboth = np.concatenate((pvc_voltage_calc, pvc_voltage_set), axis=None)
    # Iboth = np.concatenate((pvc_current_set, pvc_current_calc), axis=None)
    # # sort
    # p = np.argsort(Vboth)
    # Vlight = Vboth[p]
    # Ilight = -1 * Iboth[p]

    # fig, ax = plt.subplots()
    # ax.plot(voltage, current, ".")
    # ax.plot(Vlight, Ilight)
    # # ax.plot(Vl,Il, "--")
    # plt.show()

    # print(pvc.junction.Vth(25))
