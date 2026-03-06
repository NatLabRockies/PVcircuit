import os
from pathlib import Path

import matplotlib.pyplot as plt

# Set to True once to write baseline test files, then revert to False
REGENERATE_TEST_FILES = False
import numpy as np
import pandas as pd
import pvcircuit as pvc
import pytest
from pvcircuit import Multi2T
from pvlib import ivtools, pvsystem


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
def junction():
    return pvc.junction.Junction()


def test_junction_str(junction):

    test_file = "Junction.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(junction.__str__())

    with open(pvc.pvcpath.parent.joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(test_str, junction.__str__())

def test_junction_setter(junction):
    """
    Test the junction setters.
    """

    # test setting of n
    junction.set(n=[1, 2])
    np.testing.assert_array_equal(junction.n, np.array([1, 2]))

    # test setting of single n value
    junction.set(**{"n[0]": 3})
    np.testing.assert_array_equal(junction.n, np.array([3, 2]))

    # test mismatch when setting single n value
    with pytest.raises(IndexError, match=r"invalid junction index. Set index is 3 but junction size is 2"):
        junction.set(**{"n[2]": 4})

    # test mismatch when setting n and J0ratio of different size
    with pytest.raises(ValueError, match=r"n and J0ratio must be same size"):
        junction.set(n=[1, 2, 3], J0ratio=[1, 2])

    with pytest.raises(ValueError, match=r"n and J0ratio must be same size"):
        junction.set(n=[1, 2], J0ratio=[1, 2, 3])

    # test mismatch when setting n or J0ratio with different number of current diode number
    with pytest.raises(ValueError, match=r"setting single n or J0ratio value must match previous number of diodes"):
        junction.set(n=[1, 2, 3])

    with pytest.raises(ValueError, match=r"setting single n or J0ratio value must match previous number of diodes"):
        junction.set(J0ratio=[1, 2, 3])


    # test setting the general area with light and total area
    junction.set(area=1.23)
    np.testing.assert_almost_equal(junction.lightarea, 1.23)
    np.testing.assert_almost_equal(junction.totalarea, 1.23)

    # test setting invalid class values
    with pytest.raises(ValueError, match=r"invalid class attribute test"):
        junction.set(test=-1)
    with pytest.raises(ValueError, match=r"invalid class attribute avalanche"):
        junction.set(avalanche=1)
    with pytest.raises(ValueError, match=r"invalid class attribute mrb"):
        junction.set(mrb=1)

    # test reverse bias breakdown model keys
    junction.set(RBB="bishop")
    junction.set(avalanche=1)

    with pytest.raises(ValueError, match=r"invalid class attribute J0rb"):
        junction.set(J0rb=1)

    junction.set(RBB="JFG")
    junction.set(J0rb=1)


def test_junction_properties(junction):
    """
    Test the junction properties.
    """

    np.testing.assert_almost_equal(junction.Jphoto, 0.04)
    np.testing.assert_allclose(junction.J0, [1.3141250302231388e-15, 3.6250862475576206e-09])


def test_junction_j0init(junction):

    with pytest.raises(ValueError, match=r"J0ref and n must be same size"):
        junction._J0init(1e-15)

    junction._J0init([1e-15, 1e-9])

    np.testing.assert_allclose(junction.J0, [1e-15, 1e-9])
    np.testing.assert_allclose(junction.J0ratio, [7.609626, 2.75855506])


def test_jem(junction):

    np.testing.assert_almost_equal(junction.Jem(0.6), 1.8227873411146403e-06)
    np.testing.assert_almost_equal(junction.Jem(-0.6), 0.0)


def test_notdiode(junction):

    assert not junction.notdiode()
    junction.set(J0ratio=[0, 0])
    assert junction.notdiode()


def test_Jmultidiodes(junction):

    np.testing.assert_almost_equal(junction.Jmultidiodes(0.56), 0.00019985765193700707)


def test_JshuntRBB(junction):

    junction.set(RBB=None, Gsh=1e-4)
    np.testing.assert_almost_equal(junction.JshuntRBB(2), 0.0002)
    junction.set(RBB="JFG")
    np.testing.assert_almost_equal(junction.JshuntRBB(-3), -3.033350517003164)
    junction.set(RBB="bishop")
    np.testing.assert_almost_equal(junction.JshuntRBB(-18), -0.0018153659543416658)


def test_Vdiode(junction):

    np.testing.assert_almost_equal(junction.Vdiode(0), 0.7849550554937345)
    np.testing.assert_almost_equal(junction.Vdiode(-40e-3), 0, decimal=5)
    np.testing.assert_equal(junction.Vdiode(-41e-3), np.nan) #VLIM_REVERSE

    junction.set(RBB="bishop", Gsh=1e-4)
    np.testing.assert_almost_equal(junction.Vdiode(-41e-3), -9.652384228088378)
    junction.set(RBB="JFG")
    np.testing.assert_almost_equal(junction.Vdiode(-41e-3), -0.9224606102628319)

    junction.set(J0ratio=[0, 0])
    np.testing.assert_almost_equal(junction.Vdiode(-41e-3), 0)


def test_Vmid(junction):

    np.testing.assert_almost_equal(junction.Vmid(0), 0)
    np.testing.assert_almost_equal(junction.Vmid(0.5), 0.5)

    junction.set(Rser=0.73)
    np.testing.assert_almost_equal(junction.Vmid(0), 0.029200002624152344)
    np.testing.assert_almost_equal(junction.Vmid(0.5), 0.5291205858003947)



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

    pytest.main(['-v', __file__])

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
    # J0scale = 1000
    # Rser = resistance_series * A
    # Gsh = 1 / (resistance_shunt * A)
    # # pvc.junction.DB_PREFIX
    # Jdb = pvc.junction.Jdb(TC=TC, Eg=Eg)
    # # j0=(self.Jdb * self.J0scale)**(1./self.n) * self.J0ratio / self.J0scale
    # J0ratio = J0scale * J0ref / (Jdb * J0scale) ** (1.0 / n)

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