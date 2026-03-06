import re
from pathlib import Path

import matplotlib.pyplot as plt

# Set to True once to write baseline test files, then revert to False
REGENERATE_TEST_FILES = False
import numpy as np
import pytest

import pvcircuit as pvc
from pvcircuit import IV3T, Tandem3T


@pytest.fixture
def dev3T():
    return Tandem3T()


@pytest.fixture
def iv3t():
    return IV3T()


@pytest.fixture
def junction():
    return pvc.junction.Junction()


def test_iv3t_str(iv3t):

    test_file = "iv3t_str.txt"

    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))


def test_iv3t_setter(dev3T, iv3t):

    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)
    iv3t.convert("I", "load2dev")
    dev3T.V3T(iv3t)
    for k in iv3t.arraykeys:
        iv3t.set(**{k: 1})

    test_file = "iv3t_setter.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))


def test_dev2load(dev3T, iv3t):
    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)
    iv3t.convert("I", "load2dev")
    dev3T.V3T(iv3t)
    iv3t2 = iv3t.copy()
    iv3t.init(["IA", "IB"], ["IA", "IB"])
    iv3t.convert("I", "dev2load")
    iv3t.__str__() == iv3t2.__str__()

    iv3t.line("Vzt", 0, 1.2, 11, "Vrz", "0")


def test_line(iv3t):

    iv3t.line("Vzt", 0, 1.2, 11, "Vrz", "0")

    test_file = "iv3t_line.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))


def test_box(iv3t):

    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)

    test_file = "iv3t_box.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))


def test_nanpnt(iv3t):

    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)

    array = getattr(iv3t, "IA")
    np.testing.assert_almost_equal(array[0, 0], -25e-3)
    iv3t.nanpnt((0, 0))
    np.testing.assert_almost_equal(array[0, 0], np.nan)


def test_mpp(iv3t, dev3T):
    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)
    iv3t.convert("I", "load2dev")
    dev3T.V3T(iv3t)

    # mpp = iv3t.MPP()

    test_file = "iv3t_mpp.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))


def test_resize(iv3t, dev3T):
    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)
    iv3t.convert("I", "load2dev")
    dev3T.V3T(iv3t)

    iv3t.resize([5, 10])

    np.testing.assert_array_equal(iv3t.sizes(["VA"]), [50, 50])

    test_file = "iv3t_resize.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))


def test_calc(iv3t, dev3T):

    # Create voltage line range
    iv3t.line("Vzt", 0, 1.5, 20, "Vrz", "0")

    # Copy for comparinson
    iv3t2 = iv3t.copy()

    # Convert dev2load for voltages
    iv3t.convert("V", "dev2load")
    # Calculate current based on abs values
    dev3T.J3Tabs(iv3t)
    # convert dev2load for currents
    iv3t.convert("I", "dev2load")
    # Calculate power
    iv3t.Pcalc()
    # Check kirchhoff
    iv3t.kirchhoff(iv3t.Idevlist)
    # recalculate currents
    iv3t.convert("I", "load2dev")

    # Calculate everything fo iv3t2 directly. Should be the same?
    dev3T.I3Trel(iv3t2)
    iv3t.append(iv3t2.MPP())

    iv3t.delete(5)

    test_file = "iv3t_calc.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))


def test_copy(iv3t):
    iv3t.set(Iro=1.0, Izo=2.0, Ito=3.0, Vzt=4.0, Vrz=5.0, Vtr=6.0)
    iv3t_copy = iv3t.copy()
    np.testing.assert_array_equal(iv3t_copy.Iro, iv3t.Iro)
    np.testing.assert_array_equal(iv3t_copy.Izo, iv3t.Izo)
    np.testing.assert_array_equal(iv3t_copy.Ito, iv3t.Ito)
    np.testing.assert_array_equal(iv3t_copy.Vzt, iv3t.Vzt)
    np.testing.assert_array_equal(iv3t_copy.Vrz, iv3t.Vrz)
    np.testing.assert_array_equal(iv3t_copy.Vtr, iv3t.Vtr)
    np.testing.assert_equal(iv3t_copy.meastype, iv3t.meastype)
    np.testing.assert_equal(iv3t_copy.name, iv3t.name)
    np.testing.assert_equal(iv3t_copy.area, iv3t.area)


def test_set(iv3t):
    iv3t.set(Iro=1.0, Vrz=5.0)
    np.testing.assert_equal(iv3t.Iro, 1.0)
    np.testing.assert_equal(iv3t.Vrz, 5.0)
    with pytest.raises(ValueError):
        iv3t.set(invalid_key=123)


def test_line(iv3t):
    iv3t.line("Iro", 0.1, 1, 11, "Vrz", "-x")
    np.testing.assert_array_almost_equal(iv3t.Vrz, iv3t.Iro * -1)
    iv3t.line("Izo", 0.1, 1, 11, "Vrz", "x")
    np.testing.assert_array_almost_equal(iv3t.Vrz, iv3t.Izo * 1)


def test_box(iv3t):
    iv3t.box("Iro", 0, 10, 11, "Vrz", 0, 5, 6)
    np.testing.assert_array_almost_equal(iv3t.Iro, np.tile(np.linspace(0, 10, 11), (6, 1)))
    np.testing.assert_array_almost_equal(iv3t.Vrz, np.repeat(np.linspace(0, 5, 6).reshape(-1, 1), 11, axis=1))


def test_nanpnt(iv3t):
    iv3t.box("Iro", 0, 10, 11, "Vrz", 0, 5, 6)
    iv3t.nanpnt((1,))
    np.testing.assert_array_equal(iv3t.Iro[1], np.nan)
    np.testing.assert_array_equal(iv3t.Izo[1], np.nan)
    np.testing.assert_array_equal(iv3t.Ito[1], np.nan)


def test_sort(iv3t):
    test_data = np.arange(0, 10, 1)
    iv3t.set(Iro=np.random.choice(test_data / 1e2, len(test_data), replace=False))
    iv3t.set(Vrz=np.random.choice(test_data, len(test_data), replace=False))
    iv3t.resize(len(test_data))
    iv3t.sort("Iro")
    np.testing.assert_array_equal(iv3t.Iro, test_data / 1e2)
    iv3t.sort("Vrz")
    np.testing.assert_array_equal(iv3t.Vrz, test_data)


def test_init(iv3t):
    result = iv3t.init(["Iro", "Izo"], ["Ito"])
    np.testing.assert_equal(result, 0)
    np.testing.assert_array_equal(iv3t.Ito, iv3t.Iro)
    np.testing.assert_array_equal(iv3t.Ito, iv3t.Izo)


def test_kirchhoff(iv3t):
    iv3t.set(Vzt=10, Vrz=20)
    iv3t.kirchhoff(["Vzt", "Vrz"])
    np.testing.assert_equal(iv3t.Vtr, -30)


def test_Pcalc(iv3t):
    iv3t.set(IA=1.0, IB=2.0, VA=3.0, VB=4.0)
    iv3t.Pcalc(oper="load2dev")
    np.testing.assert_almost_equal(iv3t.Ptot, -1.0 * 3.0 - 2.0 * 4.0)


def test_loadlabel(iv3t):
    label = iv3t.loadlabel("VA")
    np.testing.assert_equal(label, "VA = Vrz")


def test_convert(iv3t):
    iv3t.set(Vrz=5.0, Vzt=10.0)
    iv3t.convert("V", "dev2load")
    np.testing.assert_almost_equal(iv3t.VA, -iv3t.Vrz)
    np.testing.assert_almost_equal(iv3t.VB, iv3t.Vtr)


def test_fromcsv():
    path = pvc.notebook_datapath
    fileA = "MS874n4papy_C_CZ_JA.csv"
    fileB = "MS874n4papy_C_CZ_JB.csv"
    iv3t = pvc.IV3T.from_csv("MS874_V_dataiv", path, fileA, fileB, "V", "CZ", area=1)  # Iscale=1000./A)

    test_file = "iv3t_fromcsv.txt"
    if REGENERATE_TEST_FILES:
        with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "w", encoding="utf8") as fout:
            fout.write(iv3t.__str__())

    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", iv3t.__str__()))

    ax1, ax2 = iv3t.plot("Iro", "Ito")


def test_plot():
    path = pvc.notebook_datapath
    fileA = "MS874n4papy_C_CZ_JA.csv"
    fileB = "MS874n4papy_C_CZ_JB.csv"
    iv3t = pvc.IV3T.from_csv("MS874_V_dataiv", path, fileA, fileB, "V", "CZ", area=1)  # Iscale=1000./A)

    ax, lst = iv3t.plot("Iro", "Ito")

    assert isinstance(ax, plt.Axes)
    assert isinstance(lst, list)
    # also test hexgrid
    np.testing.assert_equal(len(ax.lines), 2)
    iv3t.hexgrid(ax, "I", 0.1)
    np.testing.assert_equal(len(ax.lines), 62)


def generate_test_files():
    """Generate all baseline test files. Run: python tests/test_iv3t.py"""
    global REGENERATE_TEST_FILES
    REGENERATE_TEST_FILES = True

    dev3T = Tandem3T()
    iv3t = IV3T()

    print("Generating iv3t_str.txt...")
    test_iv3t_str(iv3t)
    iv3t = IV3T()
    print("Generating iv3t_setter.txt...")
    test_iv3t_setter(dev3T, iv3t)
    iv3t = IV3T()
    print("Generating iv3t_line.txt...")
    test_line(iv3t)
    iv3t = IV3T()
    print("Generating iv3t_box.txt...")
    test_box(iv3t)
    iv3t = IV3T()
    print("Generating iv3t_mpp.txt...")
    test_mpp(iv3t, dev3T)
    iv3t = IV3T()
    print("Generating iv3t_resize.txt...")
    test_resize(iv3t, dev3T)
    iv3t = IV3T()
    print("Generating iv3t_calc.txt...")
    test_calc(iv3t, dev3T)
    print("Generating iv3t_fromcsv.txt...")
    test_fromcsv()

    REGENERATE_TEST_FILES = False
    print("Done!")


if __name__ == "__main__":
    generate_test_files()
    # iv3t.Pcalc()
    # dev3T.I3Trel(iv3t)

    # v = getattr(iv3t,"VB")
    # i = getattr(iv3t,"IB")
    # plt.plot(v,i)
    # plt.show()
