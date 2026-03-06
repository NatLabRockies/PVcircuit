# -*- coding: utf-8 -*-
"""
Tests for pvcircuit.EY module
"""

import warnings

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import pytest
from scipy.integrate import trapezoid
import copy

import pvcircuit as pvc

# EY emits a DeprecationWarning on import; suppress it in tests
from pvcircuit.EY import Meteo
from pvcircuit.qe import EQET, ModelType, TemperatureModel, wvl, AM15G

_TEST_FILES = Path(__file__).parent / "test_files"
# Set to True once to write baseline test files, then revert to False
REGENERATE_TEST_FILES = False


#################################################
# Helpers
#################################################


def _load_tc_eqet():
    """Load top-cell EQET from MP905n5.csv (1st-junction columns only)."""
    path = pvc.notebook_datapath.joinpath("MP905n5.csv")
    data = pd.read_csv(path, index_col=0)
    temperatures = data.columns.to_series().str.findall(r"(\d+)C").explode().dropna().astype(int)
    junctions = np.array([int(name.split("_")[2][0]) if len(name.split("_")) > 3 else 1 for name in data.columns])
    return EQET(
        wavelength=data.index.to_numpy(),
        eqe=data.loc[:, junctions == 1].to_numpy(),
        temperature=temperatures[junctions == 1].to_numpy(),
    )


def _load_bc_eqet():
    """Load bottom-cell EQET from MP846n8.csv."""
    path = pvc.notebook_datapath.joinpath("MP846n8.csv")
    data = pd.read_csv(path, index_col=0)
    temperatures = data.columns.to_series().str.findall(r"(\d+)C").explode().dropna().astype(int)
    return EQET(
        wavelength=data.index.to_numpy(),
        eqe=data.to_numpy(),
        temperature=temperatures.to_numpy(),
    )


#################################################
# Fixtures
#################################################


@pytest.fixture(scope="module")
def tc_eqet():
    return _load_tc_eqet()


@pytest.fixture(scope="module")
def bc_eqet():
    return _load_bc_eqet()


@pytest.fixture(scope="module")
def meteo(nsrdb_data):
    wavelength, spectra_df, meteo_df = _load_nsrdb(nsrdb_data, n=10)
    return Meteo(
        wavelength,
        spectra_df,
        meteo_df["Temperature"],
        meteo_df["Wind Speed"],
        meteo_df.index,
    )


@pytest.fixture(scope="module")
def nsrdb_data():
    """
    Load the NSRDB test file once per test session.
    Returns (wavelength_nm, spectra_df, meteo_df, irradiance).
    """
    return _load_nsrdb_raw()


def _load_nsrdb_raw():
    """Load the NSRDB zip and return (wavelength_nm, spectra_df, meteo_df, irradiance)."""
    from pathlib import Path

    filepath = Path(__file__).parent / "test_files" / "2021_39p74_-105p17_one_axis.nsrdb"

    with zipfile.ZipFile(filepath) as zf:
        fname = zf.namelist()[0]
        with zf.open(fname) as f:
            meta = pd.read_csv(f, nrows=1, header=0)
        with zf.open(fname) as f:
            data = pd.read_csv(f, header=2)

    data["timestamp"] = pd.to_datetime(data[["Year", "Month", "Day", "Hour", "Minute"]], utc=True)
    tz_offset = pytz.FixedOffset(int(meta["Local Time Zone"][0] * 60))
    data["timestamp"] = data["timestamp"].dt.tz_convert(tz_offset)
    data.set_index("timestamp", inplace=True)

    meteo_full = data.iloc[:, :32].copy()
    spectra_full = data.iloc[:, 32:].copy() / 1e3  # W/m²/µm → W/m²/nm
    wavelength = (
        spectra_full.columns.str.extract(r"(\d+\.\d+)", expand=True).astype(float).to_numpy().flatten() * 1e3  # µm → nm
    )

    spectra_full.fillna(0, inplace=True)
    spectra_full[spectra_full < 0] = 0
    irradiance = trapezoid(y=spectra_full.to_numpy(), x=wavelength, axis=1)
    spectra_full.iloc[irradiance < 30] = 0

    return wavelength, spectra_full, meteo_full, irradiance


def _load_nsrdb(nsrdb_data, n=None):
    """Slice the cached NSRDB data to the first *n* daylight timesteps with irradiance >= 200 W/m^2."""
    wavelength, spectra_full, meteo_full, irradiance = nsrdb_data
    if n is not None:
        idx = np.where(irradiance >= 200)[0][:n]
        return wavelength, spectra_full.iloc[idx].copy(), meteo_full.iloc[idx].copy()
    return wavelength, spectra_full, meteo_full


#################################################
# Meteo initialisation
#################################################


def test_meteo_init(meteo):
    n = len(meteo.datetime)
    assert n == 10
    assert meteo.irradiance.shape == (n,)
    assert meteo.cell_temp.shape == (n,)
    assert float(meteo.energy_in) > 0
    assert meteo.jscs is None
    assert meteo.bandgaps is None


#################################################
# add_bandgaps / add_currents
#################################################


def test_add_bandgaps(tc_eqet, bc_eqet, meteo):
    import copy

    ey = copy.deepcopy(meteo)
    n = len(ey.cell_temp)
    cell_temps = ey.cell_temp.to_numpy()

    # fit a linear temperature model to tc bandgap
    tc_bg25, _ = tc_eqet.get_eqe_at_temperature(25).calc_Eg_Rau()
    tc_model = TemperatureModel.fit(
        tc_eqet.temperature.astype(float),
        np.array(tc_eqet.calc_Eg_Rau()[0]),
        model_types=[ModelType.LINEAR],
    )

    bc_bg25, _ = bc_eqet.get_eqe_at_temperature(25).calc_Eg_Rau()
    bc_model = TemperatureModel.fit(
        bc_eqet.temperature.astype(float),
        np.array(bc_eqet.calc_Eg_Rau()[0]),
        model_types=[ModelType.LINEAR],
    )

    ey.add_bandgaps(tc_model.apply(cell_temps, float(tc_bg25[0])))
    ey.add_bandgaps(bc_model.apply(cell_temps, float(bc_bg25[0])))

    assert ey.bandgaps.shape == (n, 2)
    # bandgaps should be physically reasonable (0.5–3 eV)
    assert np.all(ey.bandgaps > 0.5)
    assert np.all(ey.bandgaps < 3.0)

    test_file = _TEST_FILES / "ey_add_bandgaps.txt"
    if REGENERATE_TEST_FILES:
        np.savetxt(test_file, ey.bandgaps, delimiter=",")
    np.testing.assert_allclose(ey.bandgaps, np.loadtxt(test_file, delimiter=","), rtol=1e-6)


def test_add_currents(tc_eqet, bc_eqet, meteo):
    import copy

    ey = copy.deepcopy(meteo)
    n = len(ey.cell_temp)
    cell_temps = ey.cell_temp.to_numpy()  # shape (n,)

    # add spectra — tile AM1.5G to (N_wvl, n) so spectra.shape[1] == n
    tc_eqet_copy = copy.deepcopy(tc_eqet)
    bc_eqet_copy = copy.deepcopy(bc_eqet)

    spectra_tiled = np.tile(AM15G[:, np.newaxis], (1, n))  # (N_wvl, n)
    tc_eqet_copy.add_spectra(wvl, spectra_tiled)
    bc_eqet_copy.add_spectra(wvl, spectra_tiled)

    tc_currents = tc_eqet_copy.get_current_for_temperature(cell_temps, degrees=[1])
    bc_currents = bc_eqet_copy.get_current_for_temperature(cell_temps, degrees=[1])

    # filter negative
    tc_currents[tc_currents < 0] = 0
    bc_currents[bc_currents < 0] = 0

    ey.add_currents(tc_currents)
    ey.add_currents(bc_currents)

    assert ey.jscs.shape == (n, 2)
    assert np.all(ey.jscs >= 0)

    test_file = _TEST_FILES / "ey_add_jscs.txt"
    if REGENERATE_TEST_FILES:
        np.savetxt(test_file, ey.jscs, delimiter=",")
    np.testing.assert_allclose(ey.jscs, np.loadtxt(test_file, delimiter=","), rtol=1e-6)


#################################################
# get_eqe_at_temperature
#################################################


def test_get_eqe_at_temperature(tc_eqet, bc_eqet):
    # interpolate top cell to 25 °C
    # get_eqe_at_temperature returns one column per target temperature,
    # so njuncs == 1 when a scalar temperature is given.
    tc_25 = tc_eqet.get_eqe_at_temperature(25)
    assert isinstance(tc_25, EQET)
    assert tc_25.njuncs == 1
    np.testing.assert_array_equal(tc_25.wavelength, tc_eqet.wavelength)
    # EQE values should be clipped to [0, max original]
    assert np.all(tc_25.eqe >= 0)
    assert np.all(tc_25.eqe <= tc_eqet.eqe.max() + 1e-9)

    # interpolate bottom cell to 100 °C (within measurement range)
    # result has njuncs==1 regardless of the source (one column per target temp)
    bc_100 = bc_eqet.get_eqe_at_temperature(100)
    assert isinstance(bc_100, EQET)
    assert bc_100.njuncs == 1

    # unknown method raises ValueError
    with pytest.raises(ValueError, match="not implemented"):
        tc_eqet.get_eqe_at_temperature(25, method="bogus")


def test_get_eqe_at_temperature_bandgap_consistency(tc_eqet):
    """Bandgap at interpolated 25 °C should match direct 25 °C measurement."""
    # find the column index of the 25 °C measurement
    idx_25 = np.where(tc_eqet.temperature == 25)[0]
    if len(idx_25) == 0:
        pytest.skip("No 25 °C column in top-cell EQET data")

    # measured bandgap at 25 °C (first junction)
    tc_25_meas = EQET(
        tc_eqet.wavelength,
        tc_eqet.eqe[:, idx_25],
        np.array([25]),
    )
    bg_meas, _ = tc_25_meas.calc_Eg_Rau()

    # interpolated bandgap at 25 °C
    tc_25_interp = tc_eqet.get_eqe_at_temperature(25)
    bg_interp, _ = tc_25_interp.calc_Eg_Rau()

    # should agree to within 50 meV
    np.testing.assert_allclose(bg_interp[0], bg_meas[0], atol=0.05)


#################################################
# get_current_for_temperature
#################################################


def test_get_current_for_temperature_single(tc_eqet):
    import copy

    eqet = copy.deepcopy(tc_eqet)
    eqet.add_spectra()  # AM1.5G, single spectrum

    result = eqet.get_current_for_temperature([25])
    assert result.shape == (1,)
    assert result[0] > 0


def test_get_current_for_temperature_multi(bc_eqet):
    import copy

    eqet = copy.deepcopy(bc_eqet)
    n = len(eqet.temperature)
    spectra_tiled = np.tile(AM15G[:, np.newaxis], (1, n))
    eqet.add_spectra(wvl, spectra_tiled)

    result = eqet.get_current_for_temperature(eqet.temperature.astype(float), degrees=[1])
    assert result.shape == (n,)
    assert np.all(result > 0)


def test_get_current_for_temperature_no_spectra(tc_eqet):
    import copy

    eqet = copy.deepcopy(tc_eqet)
    eqet.spectra = None  # ensure spectra is cleared
    with pytest.raises(ValueError, match="Load spectral information first"):
        eqet.get_current_for_temperature([25])


#################################################
# run_ey — 2-terminal (Multi2T / CM)
#################################################


def _make_full_meteo(nsrdb_data, tc_eqet, bc_eqet, n=10):
    """Build a Meteo with bandgaps and currents populated from NSRDB data."""

    wavelength, spectra_df, meteo_df = _load_nsrdb(nsrdb_data, n=n)
    ey = Meteo(
        wavelength,
        spectra_df,
        meteo_df["Temperature"],
        meteo_df["Wind Speed"],
        meteo_df.index,
    )
    cell_temps = ey.cell_temp.to_numpy()  # shape (n,)

    tc_eqet_c = copy.deepcopy(tc_eqet)
    bc_eqet_c = copy.deepcopy(bc_eqet)

    # real spectra from NSRDB: shape (N_wvl, n)
    spectra_arr = spectra_df.to_numpy().T
    tc_eqet_c.add_spectra(wavelength, spectra_arr)
    bc_eqet_c.add_spectra(wavelength, spectra_arr)

    # currents — target temperature array length must match number of spectra columns
    tc_currents = tc_eqet_c.get_current_for_temperature(cell_temps, degrees=[1])
    bc_currents = bc_eqet_c.get_current_for_temperature(cell_temps, degrees=[1])
    tc_currents[tc_currents < 0] = 0
    bc_currents[bc_currents < 0] = 0
    ey.add_currents(tc_currents)
    ey.add_currents(bc_currents)

    # bandgaps — fit linear model to temperature-dependent bandgap
    tc_bg25, _ = tc_eqet.get_eqe_at_temperature(25).calc_Eg_Rau()
    bc_bg25, _ = bc_eqet.get_eqe_at_temperature(25).calc_Eg_Rau()
    tc_model = TemperatureModel.fit(
        tc_eqet.temperature.astype(float),
        np.array(tc_eqet.calc_Eg_Rau()[0]),
        model_types=[ModelType.LINEAR],
    )
    bc_model = TemperatureModel.fit(
        bc_eqet.temperature.astype(float),
        np.array(bc_eqet.calc_Eg_Rau()[0]),
        model_types=[ModelType.LINEAR],
    )
    ey.add_bandgaps(np.asarray(tc_model.apply(cell_temps, tc_bg25)))
    ey.add_bandgaps(np.asarray(bc_model.apply(cell_temps, bc_bg25)))

    return ey


def test_run_ey_2T(nsrdb_data, tc_eqet, bc_eqet):
    tandem2T = pvc.Multi2T()
    ey = _make_full_meteo(nsrdb_data, tc_eqet, bc_eqet, n=50)
    energy_out, ey_eff = ey.run_ey(tandem2T, "CM", multiprocessing=False)

    test_file = _TEST_FILES / "ey_run_2T_CM_n200.txt"
    if REGENERATE_TEST_FILES:
        np.savetxt(test_file, [energy_out, ey_eff], delimiter=",")
    ref = np.loadtxt(test_file, delimiter=",")
    np.testing.assert_allclose([energy_out, ey_eff], ref, rtol=1e-4)

    assert energy_out > 0
    assert 0 < ey_eff < 1


def test_run_ey_3T(nsrdb_data, tc_eqet, bc_eqet):
    tandem3T = pvc.Tandem3T()
    ey = _make_full_meteo(nsrdb_data, tc_eqet, bc_eqet, n=50)
    energy_out, ey_eff = ey.run_ey(tandem3T, "CM", multiprocessing=False)
    test_file = _TEST_FILES / "ey_run_3T_CM_n50.txt"
    if REGENERATE_TEST_FILES:
        np.savetxt(test_file, [energy_out, ey_eff], delimiter=",")
    np.testing.assert_allclose([energy_out, ey_eff], np.loadtxt(test_file, delimiter=","), rtol=1e-4)

    ey = _make_full_meteo(nsrdb_data, tc_eqet, bc_eqet, n=200)
    energy_out, ey_eff = ey.run_ey(tandem3T, "CM", multiprocessing=True)
    test_file = _TEST_FILES / "ey_run_3T_CM_n200.txt"
    if REGENERATE_TEST_FILES:
        np.savetxt(test_file, [energy_out, ey_eff], delimiter=",")
    np.testing.assert_allclose([energy_out, ey_eff], np.loadtxt(test_file, delimiter=","), rtol=1e-4)

    energy_out, ey_eff = ey.run_ey(tandem3T, "MPP", multiprocessing=True)
    test_file: Path = _TEST_FILES / "ey_run_3T_MPP_n200.txt"
    if REGENERATE_TEST_FILES:
        np.savetxt(test_file, [energy_out, ey_eff], delimiter=",")
    np.testing.assert_allclose([energy_out, ey_eff], np.loadtxt(test_file, delimiter=","), rtol=1e-4)

    energy_out, ey_eff = ey.run_ey(tandem3T, "VM-21-r", multiprocessing=True)
    test_file = _TEST_FILES / "ey_run_3T_VM21r_n200.txt"
    if REGENERATE_TEST_FILES:
        np.savetxt(test_file, [energy_out, ey_eff], delimiter=",")
    np.testing.assert_allclose([energy_out, ey_eff], np.loadtxt(test_file, delimiter=","), rtol=1e-4)

    assert energy_out > 0
    assert 0 < ey_eff < 1


def generate_test_files():
    """Generate all baseline CSV test files.

    Run this module directly to regenerate:
        python tests/test_EY.py
    """
    global REGENERATE_TEST_FILES
    REGENERATE_TEST_FILES = True

    tc_eqet = _load_tc_eqet()
    bc_eqet = _load_bc_eqet()
    nsrdb_data = _load_nsrdb_raw()

    wavelength, spectra_df, meteo_df = _load_nsrdb(nsrdb_data, n=10)
    meteo = Meteo(
        wavelength,
        spectra_df,
        meteo_df["Temperature"],
        meteo_df["Wind Speed"],
        meteo_df.index,
    )

    print("Generating ey_add_bandgaps.txt...")
    test_add_bandgaps(tc_eqet, bc_eqet, meteo)

    print("Generating ey_add_jscs.txt...")
    test_add_currents(tc_eqet, bc_eqet, meteo)

    print("Generating ey_run_2T_CM_n200.txt...")
    test_run_ey_2T(nsrdb_data, tc_eqet, bc_eqet)

    print("Generating ey_run_3T_*.txt...")
    test_run_ey_3T(nsrdb_data, tc_eqet, bc_eqet)

    REGENERATE_TEST_FILES = False
    print("Done! All baseline files generated.")


if __name__ == "__main__":
    generate_test_files()
