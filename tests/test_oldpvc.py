import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pvcircuit as pvc
from pvcircuit import Multi2T, Tandem3T

pytestmark = pytest.mark.skip(reason="legacy exact-output baselines are no longer maintained")


@pytest.fixture
def multi2T():
    return Multi2T()


@pytest.fixture
def tandem3T():
    return Tandem3T()


def test_Multi2T(multi2T):

    test_file = "oldpvc_Multi2T.txt"
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(test_str, multi2T.__str__())


def test_Multi2T_MPP(multi2T):

    test_file = "oldpvc_Multi2T_MPP.txt"

    # make dict
    multi2T_in = {}
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        for line in fin:
            data = line.rstrip().split(":")
            multi2T_in[data[0]] = float(data[1])

    assert multi2T_in == multi2T.MPP()
    np.testing.assert_equal(multi2T_in, multi2T.MPP())


def test_Multi2T_IV(multi2T):

    test_file = "oldpvc_Multi2T_IV.csv"

    multi2T_in = pd.read_csv(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), index_col=0)

    MPP = multi2T.MPP()
    voltages = np.linspace(-0.2, MPP["Voc"])
    currents = np.linspace(-0.2, MPP["Isc"])

    I2T = np.vectorize(multi2T.I2T)
    V2T = np.vectorize(multi2T.V2T)

    Vboth = np.concatenate((voltages, V2T(currents)), axis=None)
    Iboth = np.concatenate((I2T(voltages), currents), axis=None)
    # sort
    p = np.argsort(Vboth)
    Vlight = Vboth[p]
    Ilight = Iboth[p]
    multi2T_out = pd.DataFrame({"v": Vlight, "i": Ilight}).dropna()

    pd.testing.assert_series_equal(multi2T_in["v"], multi2T_out["v"])
    pd.testing.assert_series_equal(multi2T_in["i"], multi2T_out["i"])


def test_Tandem3T(tandem3T):

    test_file = "oldpvc_Tandem3T.txt"
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(test_str, tandem3T.__str__())

    # with open(os.path.join("tests", "Tandem3T.txt"), "r", encoding="utf8") as fid:
    #     tandem3T_in = [line.rstrip().strip() for line in fid]

    # for i, r in enumerate(tandem3T_in):
    #     tandem3T_out = tandem3T.__str__().split("\n")[i].strip()
    #     assert tandem3T_out in tandem3T_in
    #     # if tandem3T_out not in tandem3T_in:
    #     # print(tandem3T_in)


def test_Tandem3T_MPP(tandem3T):

    test_file = "oldpvc_Tandem3T_MPP.txt"
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", tandem3T.MPP().__str__()))


def test_Tandem3T_CM(tandem3T):

    test_file = "oldpvc_Tandem3T_CM.txt"
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        tandem3T_in = [line.rstrip().strip() for line in fin]

    for i, r in enumerate(tandem3T_in):
        tandem3T_out = tandem3T.CM().__str__().split("\n")[i].strip()
        assert tandem3T_out in tandem3T_in


def test_Tandem3T_VM21(tandem3T):

    test_file = "oldpvc_Tandem3T_VM21.txt"
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", tandem3T.VM(2, 1).__str__()))


def test_Tandem3T_VM32(tandem3T):

    test_file = "oldpvc_Tandem3T_VM32.txt"
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", tandem3T.VM(3, 2).__str__()))


def test_Tandem3T_VM32_set(tandem3T):
    tandem3T.top.set(Eg=1.87, J0ratio=[80.0, 22.0], Jext=0.0131, Gsh=1e-8, Rser=0.1)
    tandem3T.bot.set(Eg=1.419, J0ratio=[10.0, 15.0], Jext=0.0128, Gsh=5e-5, Rser=0.2, beta=3, area=0.89)
    tandem3T.set(Rz=1)

    test_file = "oldpvc_Tandem3T_VM32_set.txt"
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+", " ", test_str), re.sub(r"\s+", " ", tandem3T.VM(3, 2).__str__()))

    dev2T = Multi2T.from_3T(tandem3T)

    test_file = "oldpvc_Tandem3T_to_2Tcopy.txt"
    with open(pvc.pvcpath.parent.joinpath("tests", "test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(test_str, dev2T.__str__())


if __name__ == "__main__":
    test_Multi2T_MPP(Multi2T())
