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

    # Build the timestamp as a standalone Series and assign directly to the
    # index, avoiding a column insert on this very wide DataFrame (which
    # otherwise triggers a pandas PerformanceWarning about fragmentation).
    tz_offset = pytz.FixedOffset(int(meta["Local Time Zone"][0] * 60))
    timestamp = pd.to_datetime(data[["Year", "Month", "Day", "Hour", "Minute"]], utc=True).dt.tz_convert(tz_offset)
    data.index = pd.DatetimeIndex(timestamp, name="timestamp")

    meteo_full = data.iloc[:, :32].copy()
    spectra_full = data.iloc[:, 32:].copy() / 1e3  # W/m^2/\mum -> W/m^2/nm
    wavelength = (
        spectra_full.columns.str.extract(r"(\d+\.\d+)", expand=True).astype(float).to_numpy().flatten() * 1e3  # \mum -> nm
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
    # bandgaps should be physically reasonable (0.5-3 eV)
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

    # add spectra -- tile AM1.5G to (N_wvl, n) so spectra.shape[1] == n
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
    # interpolate top cell to 25 \degC
    # get_eqe_at_temperature returns one column per target temperature,
    # so njuncs == 1 when a scalar temperature is given.
    tc_25 = tc_eqet.get_eqe_at_temperature(25)
    assert isinstance(tc_25, EQET)
    assert tc_25.njuncs == 1
    np.testing.assert_array_equal(tc_25.wavelength, tc_eqet.wavelength)
    # EQE values should be clipped to [0, max original]
    assert np.all(tc_25.eqe >= 0)
    assert np.all(tc_25.eqe <= tc_eqet.eqe.max() + 1e-9)

    # interpolate bottom cell to 100 \degC (within measurement range)
    # result has njuncs==1 regardless of the source (one column per target temp)
    bc_100 = bc_eqet.get_eqe_at_temperature(100)
    assert isinstance(bc_100, EQET)
    assert bc_100.njuncs == 1

    # unknown method raises ValueError
    with pytest.raises(ValueError, match="not implemented"):
        tc_eqet.get_eqe_at_temperature(25, method="bogus")


def test_get_eqe_at_temperature_bandgap_consistency(tc_eqet):
    r"""Bandgap at interpolated 25 \degC should match direct 25 \degC measurement."""
    # find the column index of the 25 \degC measurement
    idx_25 = np.where(tc_eqet.temperature == 25)[0]
    if len(idx_25) == 0:
        pytest.skip("No 25 degC column in top-cell EQET data")

    # measured bandgap at 25 \degC (first junction)
    tc_25_meas = EQET(
        tc_eqet.wavelength,
        tc_eqet.eqe[:, idx_25],
        np.array([25]),
    )
    bg_meas, _ = tc_25_meas.calc_Eg_Rau()

    # interpolated bandgap at 25 \degC
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
# run_ey -- 2-terminal (Multi2T / CM)
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

    # currents -- target temperature array length must match number of spectra columns
    tc_currents = tc_eqet_c.get_current_for_temperature(cell_temps, degrees=[1])
    bc_currents = bc_eqet_c.get_current_for_temperature(cell_temps, degrees=[1])
    tc_currents[tc_currents < 0] = 0
    bc_currents[bc_currents < 0] = 0
    ey.add_currents(tc_currents)
    ey.add_currents(bc_currents)

    # bandgaps -- fit linear model to temperature-dependent bandgap
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
    # the timestep-by-timestep solver must stay on the same baseline
    np.testing.assert_allclose(ey.run_ey(tandem2T, "CM", multiprocessing=False, vectorized=False), ref, rtol=1e-4)

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
    # the timestep-by-timestep solver must stay on the same baseline
    np.testing.assert_allclose(ey.run_ey(tandem3T, "CM", multiprocessing=False, vectorized=False), np.loadtxt(test_file, delimiter=","), rtol=1e-4)

    ey = _make_full_meteo(nsrdb_data, tc_eqet, bc_eqet, n=200)
    energy_out, ey_eff = ey.run_ey(tandem3T, "CM", multiprocessing=True)
    test_file = _TEST_FILES / "ey_run_3T_CM_n200.txt"
    if REGENERATE_TEST_FILES:
        np.savetxt(test_file, [energy_out, ey_eff], delimiter=",")
    np.testing.assert_allclose([energy_out, ey_eff], np.loadtxt(test_file, delimiter=","), rtol=1e-4)
    # ... and so must its multiprocessing pool
    np.testing.assert_allclose(ey.run_ey(tandem3T, "CM", multiprocessing=True, vectorized=False), np.loadtxt(test_file, delimiter=","), rtol=1e-4)

    energy_out, ey_eff = ey.run_ey(tandem3T, "MPP", multiprocessing=True)
    test_file: Path = _TEST_FILES / "ey_run_3T_MPP_n200.txt"
    if REGENERATE_TEST_FILES:
        np.savetxt(test_file, [energy_out, ey_eff], delimiter=",")
    np.testing.assert_allclose([energy_out, ey_eff], np.loadtxt(test_file, delimiter=","), rtol=1e-4)
    # Tandem3T.MPP stops on a coarse current grid: timestep by timestep the yield is up to 1e-3 lower
    per_row = ey.run_ey(tandem3T, "MPP", multiprocessing=False, vectorized=False)
    np.testing.assert_allclose(per_row, np.loadtxt(test_file, delimiter=","), rtol=1e-3)
    assert per_row[0] <= energy_out

    energy_out, ey_eff = ey.run_ey(tandem3T, "VM-21-r", multiprocessing=True)
    test_file = _TEST_FILES / "ey_run_3T_VM21r_n200.txt"
    if REGENERATE_TEST_FILES:
        np.savetxt(test_file, [energy_out, ey_eff], delimiter=",")
    np.testing.assert_allclose([energy_out, ey_eff], np.loadtxt(test_file, delimiter=","), rtol=1e-4)

    energy_out, ey_eff = ey.run_ey(tandem3T, "VM-21-s", multiprocessing=True)
    test_file = _TEST_FILES / "ey_run_3T_VM21s_n200.txt"
    if REGENERATE_TEST_FILES:
        np.savetxt(test_file, [energy_out, ey_eff], delimiter=",")
    np.testing.assert_allclose([energy_out, ey_eff], np.loadtxt(test_file, delimiter=","), rtol=1e-4)

    assert energy_out > 0
    assert 0 < ey_eff < 1


#################################################
# Module-level helper functions
#################################################


def test_VMlist():
    """VMlist generates the documented ordered list of VM configurations."""
    from pvcircuit.EY import VMlist

    result = VMlist(3)
    # Always starts with the two non-VM configs in this order
    assert result[0] == "MPP"
    assert result[1] == "CM"
    # All VM entries follow the VMmn convention with m > n >= 1
    vm_entries = result[2:]
    for entry in vm_entries:
        assert entry.startswith("VM")
        assert len(entry) == 4
        m, n = int(entry[2]), int(entry[3])
        assert m > n >= 1
        # No common factor of 2/3/5 (coprime under these primes per implementation)
        for prime in (2, 3, 5):
            assert not (m % prime == 0 and n % prime == 0)

    # Specific known values for mmax=3
    assert "VM21" in result
    assert "VM31" in result
    assert "VM32" in result

    # mmax > 9 raises
    with pytest.raises(ValueError, match="smaller than 10"):
        VMlist(10)


def test_VMloss_known_configs():
    """VMloss returns documented loss factors for canonical operations."""
    from pvcircuit.EY import VMloss

    tandem3T = pvc.Tandem3T()
    multi2T = pvc.Multi2T()

    # Multi2T always returns 1 regardless of oper
    assert VMloss(multi2T, "CM", 6) == 1
    assert VMloss(multi2T, "MPP", 12) == 1

    # Tandem3T: MPP and CM both return 1
    assert VMloss(tandem3T, "MPP", 6) == 1
    assert VMloss(tandem3T, "CM", 6) == 1

    # VM-21-r: endloss = max(2,1)-1 = 1, lossfactor = 1 - 1/6 = 5/6
    np.testing.assert_allclose(VMloss(tandem3T, "VM-21-r", 6), 5 / 6)
    # VM-21-s: endloss = (2+1)-1 = 2, lossfactor = 1 - 2/6 = 4/6
    np.testing.assert_allclose(VMloss(tandem3T, "VM-21-s", 6), 4 / 6)
    # VM-32-r: endloss = max(3,2)-1 = 2, lossfactor = 1 - 2/10 = 8/10
    np.testing.assert_allclose(VMloss(tandem3T, "VM-32-r", 10), 8 / 10)
    # Floor at 0 when ncells too small: endloss=2 > ncells=1 -> 0
    assert VMloss(tandem3T, "VM-32-r", 1) == 0

    # Malformed VM operation raises
    with pytest.raises(ValueError, match="VM-"):
        VMloss(tandem3T, "VM-21", 6)
    # Invalid r/s suffix raises (function falls through; endloss undefined)
    with pytest.raises((ValueError, UnboundLocalError)):
        VMloss(tandem3T, "VM-21-x", 6)


def test_sandia_T_pvlib_parity():
    """pvcircuit.EY.sandia_T matches pvlib's open_rack_cell_polymerback model."""
    from pvcircuit.EY import sandia_T

    # Parameters hard-coded in sandia_T (open_rack_cell_polymerback)
    a, b, deltaT = -3.56, -0.075, 3.0

    poa = np.array([100.0, 500.0, 800.0, 1000.0])
    wind = np.array([0.0, 1.0, 2.0, 5.0])
    tair = np.array([10.0, 25.0, 30.0, 35.0])

    # Hand-roll the Sandia SAPM cell temperature formula
    temp_module = poa * np.exp(a + b * wind) + tair
    expected = temp_module + (poa / 1000.0) * deltaT

    np.testing.assert_allclose(sandia_T(poa, wind, tair), expected, rtol=1e-12)

    # Scalar inputs work too
    np.testing.assert_allclose(
        sandia_T(1000.0, 0.0, 25.0),
        25.0 + 1000.0 * np.exp(a) + 3.0,
        rtol=1e-12,
    )


def test_meteo_filter_methods(meteo):
    """filter_ape, filter_spectra, filter_custom, calc_ape, reindex."""
    ey = copy.deepcopy(meteo)
    n = len(ey.datetime)

    # calc_ape populates average_photon_energy
    assert ey.average_photon_energy is None
    ey.calc_ape()
    assert ey.average_photon_energy is not None
    assert len(ey.average_photon_energy) == n
    # APE for daylight solar spectra is typically ~1.5 - 2.0 eV
    finite_ape = ey.average_photon_energy[np.isfinite(ey.average_photon_energy)]
    assert np.all(finite_ape > 1.0)
    assert np.all(finite_ape < 3.0)

    # filter_ape returns a new Meteo, keeps points in range
    n_full = len(ey.spectra)
    filt_ape = ey.filter_ape(min_ape=1.0, max_ape=3.0)
    assert filt_ape is not ey
    assert len(filt_ape.spectra) == len(filt_ape.irradiance) == len(filt_ape.cell_temp)
    assert len(filt_ape.spectra) <= n_full

    # Restrictive filter drops everything
    filt_empty = ey.filter_ape(min_ape=10.0, max_ape=20.0)
    assert len(filt_empty.spectra) == 0

    # filter_spectra: with default (0, 10) keeps all NSRDB rows (W/m^2/nm well below 10)
    filt_spec = ey.filter_spectra(min_spectra=0, max_spectra=10)
    assert filt_spec is not ey
    assert len(filt_spec.spectra) == len(filt_spec.irradiance) == len(filt_spec.cell_temp)
    assert len(filt_spec.spectra) <= n_full

    # filter_custom with boolean mask
    mask = np.zeros(n, dtype=bool)
    mask[: n // 2] = True  # keep first half
    filt_custom = ey.filter_custom(mask)
    assert filt_custom is not ey
    assert len(filt_custom.spectra) == n // 2

    # reindex: shift datetime by a value smaller than tolerance returns matched data,
    # larger than tolerance returns NaN-filled rows.
    new_idx = ey.datetime + pd.Timedelta(seconds=10)
    reidx = ey.reindex(new_idx, method="nearest", tolerance=pd.Timedelta(seconds=30))
    assert reidx is not ey
    assert len(reidx.spectra) == n


#################################################
# Regression tests for the 2026-08 EY audit
#################################################


def _synthetic_meteo(ndays=2, freq="h", start="2020-06-01"):
    """Regular hourly timeline with a sinusoidal day and zero-irradiance nights."""
    t = pd.date_range(start, periods=24 * ndays, freq=freq)
    wl = np.linspace(280, 4000, 373)
    hour = t.hour.values
    irr = np.where((hour >= 6) & (hour <= 18), np.sin(np.pi * (hour - 6) / 12) * 1000, 0.0)  # W/m^2
    spec = pd.DataFrame(np.outer(irr, np.ones_like(wl) / (wl[-1] - wl[0])), index=t)
    m = Meteo(wl, spec, pd.Series(20.0, index=t), pd.Series(1.0, index=t), t)
    return m, irr, t


def _populate(m, irr, jsc_top=20.0, jsc_bot=18.0):
    m.add_currents(irr / 1000 * jsc_top)
    m.add_currents(irr / 1000 * jsc_bot)
    m.add_bandgaps(np.full(len(irr), 1.7))
    m.add_bandgaps(np.full(len(irr), 1.1))
    return m


def test_time_weights_match_trapezoid():
    from pvcircuit.EY import _time_weights

    t = pd.date_range("2020-01-01", periods=30, freq="h")
    y = np.random.default_rng(0).random(30)
    w = _time_weights(t)
    np.testing.assert_allclose(np.sum(y * w), trapezoid(y, (t - t[0]).total_seconds()))
    # irregular timeline
    sel = [0, 1, 2, 5, 6, 10, 20, 29]
    t2 = t[sel]
    y2 = y[sel]
    np.testing.assert_allclose(np.sum(y2 * _time_weights(t2)), trapezoid(y2, (t2 - t2[0]).total_seconds()))
    assert len(_time_weights(t[:1])) == 1 and len(_time_weights(t[:0])) == 0


def test_run_ey_2T_night_rows_not_nan():
    """Multi2T.MPP() returns NaN at zero photocurrent; the yield must still be finite."""
    m, irr, _ = _synthetic_meteo(ndays=1)
    _populate(m, irr)
    energy_out, eff = m.run_ey(pvc.Multi2T(), "CM", multiprocessing=False)
    assert np.isfinite(energy_out) and energy_out > 0
    assert 0 < eff < 1
    assert np.all(np.isfinite(m.results.to_numpy()))
    # night rows give exactly zero output
    assert np.all(m.results["Pmp"].to_numpy()[irr == 0] == 0)
    # and the same device as Tandem3T CM gives (nearly) the same yield
    m3, irr3, _ = _synthetic_meteo(ndays=1)
    _populate(m3, irr3)
    energy_out_3T, _ = m3.run_ey(pvc.Tandem3T(), "CM", multiprocessing=False)
    np.testing.assert_allclose(energy_out, energy_out_3T, rtol=2e-2)


def test_run_ey_nan_jsc_rows_zero_output():
    m, irr, _ = _synthetic_meteo(ndays=1)
    jsc = irr / 1000 * 20.0
    jsc[12] = np.nan  # noon row missing
    m.add_currents(jsc)
    m.add_currents(irr / 1000 * 18.0)
    m.add_bandgaps(np.full(len(irr), 1.7))
    m.add_bandgaps(np.full(len(irr), 1.1))
    energy_out, _ = m.run_ey(pvc.Tandem3T(), "CM", multiprocessing=False)
    assert np.isfinite(energy_out)
    assert m.results["Pmp"].iloc[12] == 0


def test_filtered_meteo_energy_in_no_gap_bridging():
    """Dropping night rows must not let the integral bridge sunset -> sunrise."""
    m, irr, _ = _synthetic_meteo(ndays=10)
    e_full = m.energy_in
    day = irr > 0
    e_day = np.sum(irr[day] * m.dt.to_numpy()[day]) / 3600 / 1000

    mf = m.filter_ape()  # drops the all-zero (night) rows
    assert len(mf.datetime) == day.sum()
    np.testing.assert_allclose(mf.energy_in, e_day, rtol=1e-12)
    # previously: trapezoid over the gappy timeline inflated this by ~18 %
    assert mf.energy_in <= e_full * (1 + 1e-12)

    keep = irr > 300  # drops the 07:00 / 17:00 rows (259 W/m^2) as well as night
    mc = m.filter_custom(m.irradiance.to_numpy() > 300)
    np.testing.assert_allclose(mc.energy_in, np.sum(irr[keep] * m.dt.to_numpy()[keep]) / 3600 / 1000)
    assert mc.energy_in < e_full

    # yield on the filtered copy equals the sum over the same rows of the full run
    _populate(m, irr)
    m.run_ey(pvc.Tandem3T(), "CM", multiprocessing=False)
    mf2 = m.filter_custom(keep)
    e_out_filtered, _ = mf2.run_ey(pvc.Tandem3T(), "CM", multiprocessing=False)
    e_out_full_rows = np.sum(m.results["Pmp"].to_numpy()[keep] / pvc.Tandem3T().totalarea * m.dt.to_numpy()[keep]) / 3.6e3 / 1e3 * 1e4
    np.testing.assert_allclose(e_out_filtered, e_out_full_rows, rtol=1e-6)


def test_run_ey_column_count_check():
    m, irr, _ = _synthetic_meteo(ndays=1)
    for k in range(3):
        m.add_currents(irr / 1000 * (20 - k))  # 3 columns
    m.add_bandgaps(np.full(len(irr), 1.7))
    m.add_bandgaps(np.full(len(irr), 1.1))
    with pytest.raises(ValueError, match="jscs has 3 columns"):
        m.run_ey(pvc.Tandem3T(), "CM", multiprocessing=False)
    with pytest.raises(ValueError, match="jscs has 3 columns"):
        m.run_ey(pvc.Multi2T(), "CM", multiprocessing=False)
    m2, irr2, _ = _synthetic_meteo(ndays=1)
    m2.add_currents(irr2 / 1000 * 20)
    m2.add_bandgaps(np.full(len(irr2), 1.7))
    m2.add_bandgaps(np.full(len(irr2), 1.1))
    with pytest.raises(ValueError, match="jscs has 1 columns"):
        m2.run_ey(pvc.Tandem3T(), "CM", multiprocessing=False)
    m3, _, _ = _synthetic_meteo(ndays=1)
    with pytest.raises(ValueError, match="add_currents"):
        m3.run_ey(pvc.Tandem3T(), "CM", multiprocessing=False)


def test_meteo_input_validation():
    m, _, t = _synthetic_meteo(ndays=1)
    wl = m.wavelength
    with pytest.raises(ValueError, match="row-aligned"):
        Meteo(wl, m.spectra, pd.Series(20.0, index=t[:-1]), pd.Series(1.0, index=t), t)
    with pytest.raises(ValueError, match="spectra must be"):
        Meteo(wl[:-1], m.spectra, pd.Series(20.0, index=t), pd.Series(1.0, index=t), t)
    # a RangeIndex on the spectra is fine (positional alignment)
    m2 = Meteo(wl, m.spectra.reset_index(drop=True), pd.Series(20.0, index=t), pd.Series(1.0, index=t), t)
    assert len(m2.datetime) == len(t)
    np.testing.assert_allclose(m2.energy_in, m.energy_in)


def test_meteo_retains_uncomputable_rows_as_zero_output():
    m, irr, t = _synthetic_meteo(ndays=1)
    spectra = m.spectra.copy()
    temperature = pd.Series(20.0, index=t)
    wind = pd.Series(1.0, index=t)
    spectra.iloc[5, 0] = np.nan
    spectra.iloc[6, 1] = np.inf
    temperature.iloc[12] = np.nan
    wind.iloc[13] = np.inf

    retained = Meteo(m.wavelength, spectra, temperature, wind, t)

    assert retained.datetime.equals(t)
    assert retained.spectra.iloc[5, 0] == 0.0
    assert retained.spectra.iloc[6, 1] == 0.0
    assert np.isnan(retained.cell_temp.iloc[12])
    assert not np.isfinite(retained.cell_temp.iloc[13])
    np.testing.assert_allclose(retained.dt.to_numpy(), m.dt.to_numpy())

    _populate(retained, irr)
    retained.run_ey(pvc.Tandem3T(), "CM", multiprocessing=False)
    assert retained.results is not None
    assert retained.results["Pmp"].iloc[12] == 0.0
    assert retained.results["Pmp"].iloc[13] == 0.0


def test_reindex_aligns_numpy_arrays_and_energy_in():
    m, irr, t = _synthetic_meteo(ndays=2)
    _populate(m, irr)
    m.calc_ape()
    new_idx = t[::2] + pd.Timedelta(seconds=10)
    r = m.reindex(new_idx, method="nearest", tolerance=pd.Timedelta(seconds=30))
    assert len(r.datetime) == len(r.cell_temp) == len(r.spectra) == len(r.dt) == len(new_idx)
    assert r.jscs.shape == (len(new_idx), 2)
    assert r.bandgaps.shape == (len(new_idx), 2)
    assert len(r.average_photon_energy) == len(new_idx)
    np.testing.assert_allclose(r.jscs[:, 0], m.jscs[::2, 0])
    # 2-hourly weights -> trapezoid on the coarser grid
    np.testing.assert_allclose(r.energy_in, trapezoid(irr[::2], (t[::2] - t[0]).total_seconds()) / 3600 / 1000)
    # unmatched rows become NaN and are excluded from energy_in
    far = t[::2] + pd.Timedelta(minutes=20)
    r2 = m.reindex(far, method="nearest", tolerance=pd.Timedelta(seconds=30))
    assert np.all(np.isnan(r2.jscs))
    assert r2.energy_in == 0.0
    energy_out, _ = r.run_ey(pvc.Tandem3T(), "CM", multiprocessing=False)
    assert np.isfinite(energy_out)


def test_tandem3T_set_per_junction():
    d = pvc.Tandem3T()
    d.set(n=[[1.5], [1.0]], J0ratio=[[10.0], [20.0]])
    np.testing.assert_array_equal(d.top.n, [1.5])
    np.testing.assert_array_equal(d.bot.n, [1.0])
    d.set(Rser=[1.0, 2.0])
    assert d.top.Rser == 1.0 and d.bot.Rser == 2.0
    d2 = pvc.Tandem3T()
    d2.set(n=[1.5, 1.0], J0ratio=[10, 20])  # flat: same two-diode model in both junctions
    np.testing.assert_array_equal(d2.top.n, [1.5, 1.0])
    np.testing.assert_array_equal(d2.bot.n, [1.5, 1.0])


def test_temperature_model_stores_ref_value(tc_eqet):
    eg_T = np.array(tc_eqet.calc_Eg_Rau()[0])
    model = TemperatureModel.fit(tc_eqet.temperature.astype(float), eg_T, model_types=[ModelType.LINEAR], Tref=25)
    assert model.Tref == 25
    assert model.ref_value is not None
    np.testing.assert_allclose(float(model.apply(25.0)), model.ref_value)
    np.testing.assert_allclose(model.apply(np.array([10.0, 40.0])), model.apply(np.array([10.0, 40.0]), model.ref_value))
    with pytest.raises(ValueError, match="no ref_value"):
        TemperatureModel(ModelType.LINEAR, model.params).apply(25.0)


#################################################
# VM cell type and 2T short circuit of a Tandem3T
#################################################


def _device_at_timestep(m, k, model):
    """Copy of ``model`` at the conditions of timestep ``k`` of Meteo ``m``."""
    dev = model.copy()
    dev.top.set(Eg=m.bandgaps[k, 0], sigma=0.0, Jext=m.jscs[k, 0] / 1e3, TC=m.cell_temp.iloc[k])
    dev.bot.set(Eg=m.bandgaps[k, 1], sigma=0.0, Jext=m.jscs[k, 1] / 1e3, TC=m.cell_temp.iloc[k])
    return dev


def _vm_power(m, k, equal_pn):
    dev = _device_at_timestep(m, k, pvc.Tandem3T())
    dev.bot.pn = dev.top.pn if equal_pn else -dev.top.pn
    _, mpp = dev.VM(2, 1)
    return float(mpp.Ptot[0])


@pytest.mark.parametrize("vectorized, rtol", [(False, 1e-9), (True, 2e-5)])
@pytest.mark.parametrize("suffix, equal_pn", [("r", True), ("s", False)])
def test_run_ey_VM_cell_type_follows_label(suffix, equal_pn, vectorized, rtol):
    """'VM-..-r' must solve an r-type cell (equal pn), 'VM-..-s' an s-type cell (opposite pn).

    pvcircuit convention: Tandem3T defaults to pn = [-1, 1] (s-type); the
    r-type tests and notebooks set bot.pn equal to top.pn.
    """
    m, irr, _ = _synthetic_meteo(ndays=1)
    _populate(m, irr)
    m.run_ey(pvc.Tandem3T(), f"VM-21-{suffix}", multiprocessing=False, vectorized=vectorized)
    for k in (9, 12, 15):
        np.testing.assert_allclose(m.results["Pmp"].iloc[k], _vm_power(m, k, equal_pn), rtol=rtol)
        # the two cell types give different power, so the label matters
        assert abs(_vm_power(m, k, True) / _vm_power(m, k, False) - 1.0) > 1e-3


def test_run_ey_VM_rejects_unknown_cell_type():
    m, irr, _ = _synthetic_meteo(ndays=1)
    _populate(m, irr)
    with pytest.raises(ValueError, match="VM-"):
        m.run_ey(pvc.Tandem3T(), "VM-21-x", multiprocessing=False)


def _leaky_tandem3T():
    """s-type tandem with shunts and series resistance: Isc3 and the 2T short circuit differ."""
    dev = pvc.Tandem3T()
    dev.top.set(Gsh=5e-4, Rser=1.25)
    dev.bot.set(Gsh=5e-4, Rser=1.25)
    return dev


def test_run_ey_3T_CM_reports_2T_short_circuit():
    """CM leaves the z contact floating: Isc is the series-stack value (Vtr = 0, Izo = 0).

    Isc3 shorts all three terminals, so the z contact carries the mismatch
    current and min(|IA|, |IB|) is only the smaller subcell current.
    """
    m, irr, _ = _synthetic_meteo(ndays=1)
    _populate(m, irr)  # 20 / 18 mA/cm^2 at noon: bottom limited
    m.run_ey(_leaky_tandem3T(), "CM", multiprocessing=False)
    for k in (9, 12):
        dev = _device_at_timestep(m, k, _leaky_tandem3T())
        series = pvc.Multi2T.from_3T(dev).Isc()  # resolves 1e-7 of the limiting current
        isc3 = dev.Isc3()
        three_terminal = min(abs(isc3.IA[0]), abs(isc3.IB[0]))
        np.testing.assert_allclose(m.results["Isc"].iloc[k], series, rtol=1e-6)
        assert abs(three_terminal / series - 1.0) > 1e-2


def test_run_ey_3T_r_type_CM_keeps_Isc3():
    """An r-type cell has no 2T series operation; its CM row keeps the Isc3 value."""
    r_type = pvc.Tandem3T()
    r_type.bot.set(pn=-1)
    m, irr, _ = _synthetic_meteo(ndays=1)
    _populate(m, irr)
    m.run_ey(r_type, "CM", multiprocessing=False)
    dev = _device_at_timestep(m, 12, r_type)
    isc3 = dev.Isc3()
    np.testing.assert_allclose(m.results["Isc"].iloc[12], min(abs(isc3.IA[0]), abs(isc3.IB[0])), rtol=1e-9)


#################################################
# run_ey solves all timesteps at once (the *_rows methods of the device)
#################################################


def _run_both(make_model, oper="CM", ndays=1):
    """run_ey with all timesteps at once and timestep by timestep on the same synthetic day(s).

    ``irr > 1.0`` selects the lit rows: the 18:00 sample of the synthetic day is
    a rounding-level 1e-13 W/m^2, which is dark for the solver.
    """
    out = {}
    for vectorized in (True, False):
        m, irr, _ = _synthetic_meteo(ndays=ndays)
        _populate(m, irr)
        energy_out, _ = m.run_ey(make_model(), oper, multiprocessing=False, vectorized=vectorized)
        out[vectorized] = (energy_out, m.results.copy())
    return out[True], out[False], irr


def _reversed_polarity_tandem3T():
    return pvc.Tandem3T(pn=[1, -1])


def _unequal_area_tandem3T():
    dev = pvc.Tandem3T()
    dev.top.set(totalarea=0.8, lightarea=0.8, Rser=0.5)
    dev.bot.set(lightarea=0.9, Rser=0.5)
    return dev


def _r_type_tandem3T():
    dev = pvc.Tandem3T()
    dev.bot.set(pn=-1)
    return dev


# The per-row searches stop on a grid: Multi2T.MPP / Tandem3T.CM / Tandem3T.VM are
# up to ~2e-5 low on a single row, Tandem3T.MPP (two loads) up to ~1e-3. The
# *_rows methods refine to the optimum.
_ALL_ROWS_CASES = [
    (pvc.Multi2T, "CM", 5e-5, 2e-3),
    (pvc.Tandem3T, "CM", 5e-5, 2e-3),
    (_reversed_polarity_tandem3T, "CM", 5e-5, 2e-3),
    (_leaky_tandem3T, "CM", 5e-5, 2e-3),
    (_unequal_area_tandem3T, "CM", 5e-5, 2e-3),
    (pvc.Tandem3T, "MPP", 2e-3, 5e-2),
    (_reversed_polarity_tandem3T, "MPP", 2e-3, 5e-2),
    (_leaky_tandem3T, "MPP", 2e-3, 5e-2),
    (_unequal_area_tandem3T, "MPP", 2e-3, 5e-2),
    (pvc.Tandem3T, "VM-21-s", 5e-5, 1e-2),
    (pvc.Tandem3T, "VM-21-r", 5e-5, 1e-2),
    (pvc.Tandem3T, "VM-32-r", 5e-5, 1e-2),
    (_reversed_polarity_tandem3T, "VM-21-r", 5e-5, 1e-2),
    (_leaky_tandem3T, "VM-11-s", 5e-5, 1e-2),
    (_unequal_area_tandem3T, "VM-21-s", 5e-5, 1e-2),
]


@pytest.mark.parametrize("make_model, oper, rtol_power, rtol_point", _ALL_ROWS_CASES)
def test_run_ey_vectorized_matches_per_row(make_model, oper, rtol_power, rtol_point):
    (energy_vec, vec), (energy_row, row), irr = _run_both(make_model, oper)
    np.testing.assert_allclose(energy_vec, energy_row, rtol=rtol_power)
    np.testing.assert_allclose(vec["Pmp"], row["Pmp"], rtol=rtol_power)
    assert np.all(vec["Pmp"] >= row["Pmp"] * (1.0 - 1e-9)), "the refined optimum must not be below the grid optimum"
    np.testing.assert_allclose(vec["Voc"], row["Voc"], rtol=1e-9)
    np.testing.assert_allclose(vec["Isc"], row["Isc"], rtol=1e-6)
    np.testing.assert_allclose(vec["Vmp"], row["Vmp"], rtol=rtol_point)
    np.testing.assert_allclose(vec["Imp"], row["Imp"], rtol=rtol_point)
    assert np.all(vec.to_numpy()[irr == 0] == 0.0)
    assert np.all(vec["Pmp"].to_numpy()[irr > 1.0] > 0.0)


def test_run_ey_vectorized_keeps_voltage_sign_of_reversed_polarity():
    """The per-row solver reports VA - VB, which is negative for pn = [1, -1]."""
    for oper in ("CM", "MPP", "VM-21-s"):
        (_, vec), (_, row), irr = _run_both(_reversed_polarity_tandem3T, oper)
        assert np.all(row["Voc"].to_numpy()[irr > 1.0] < 0.0)
        assert np.all(vec["Voc"].to_numpy()[irr > 1.0] < 0.0)
        assert np.all(vec["Vmp"].to_numpy()[irr > 1.0] < 0.0)


@pytest.mark.parametrize("make_model, oper", [(pvc.Multi2T, "CM"), (pvc.Tandem3T, "CM"), (pvc.Tandem3T, "MPP"), (pvc.Tandem3T, "VM-21-r"), (pvc.Tandem3T, "VM-21-s")])
def test_run_ey_solves_all_timesteps_at_once_by_default(monkeypatch, make_model, oper):
    def not_allowed(*args, **kwargs):
        raise AssertionError("unexpected solver")

    m, irr, _ = _synthetic_meteo(ndays=1)
    _populate(m, irr)
    model = make_model()
    before = str(model)
    with monkeypatch.context() as patch:
        patch.setattr(pvc.EY, "_calc_yield_async", not_allowed)
        energy_vec, _ = m.run_ey(model, oper)  # default: no pool, no per-row call
    with monkeypatch.context() as patch:
        patch.setattr(pvc.EY, "_calc_yield_rows", not_allowed)
        energy_row, _ = m.run_ey(model, oper, multiprocessing=False, vectorized=False)
    np.testing.assert_allclose(energy_vec, energy_row, rtol=2e-3)
    assert str(model) == before, "run_ey must not modify the device"


def test_run_ey_solves_an_r_type_CM_timestep_by_timestep(monkeypatch):
    """An r-type cell in CM operation is not a series stack: it has no *_rows solver."""

    def not_allowed(*args, **kwargs):
        raise AssertionError("an r-type cell in CM operation must not be solved by the *_rows methods")

    monkeypatch.setattr(pvc.EY, "_calc_yield_rows", not_allowed)
    m, irr, _ = _synthetic_meteo(ndays=1)
    _populate(m, irr)
    energy_out, _ = m.run_ey(_r_type_tandem3T(), "CM", multiprocessing=False)
    assert np.isfinite(energy_out)


def test_run_ey_legacy_solver_setting(monkeypatch):
    """junction.SOLVER = "legacy" restores the timestep-by-timestep run, also in the worker processes."""

    def not_allowed(*args, **kwargs):
        raise AssertionError("the legacy setting must not use the *_rows methods")

    m, irr, _ = _synthetic_meteo(ndays=1)
    _populate(m, irr)
    m.run_ey(pvc.Multi2T(), "CM", multiprocessing=False, vectorized=False)
    fast = m.results.copy()
    monkeypatch.setattr(pvc.junction, "SOLVER", "legacy")
    with monkeypatch.context() as patch:
        patch.setattr(pvc.EY, "_calc_yield_rows", not_allowed)
        m.run_ey(pvc.Multi2T(), "CM", multiprocessing=False)
        legacy = m.results.copy()
        m.run_ey(pvc.Multi2T(), "CM", multiprocessing=True)
        pool = m.results.copy()
    np.testing.assert_array_equal(pool.to_numpy(), legacy.to_numpy())  # the workers use the legacy solver, too
    lit = irr > 1.0
    # the legacy Multi2T.Isc steps to 1e-7 of the root: close to the Newton result, but not it
    np.testing.assert_allclose(legacy["Isc"], fast["Isc"], rtol=1e-6)
    assert np.all(legacy["Isc"].to_numpy()[lit] != fast["Isc"].to_numpy()[lit])
    # an explicit request still solves all timesteps at once
    m.run_ey(pvc.Multi2T(), "CM", vectorized=True)
    np.testing.assert_allclose(m.results["Pmp"], legacy["Pmp"], rtol=1e-6)


@pytest.mark.parametrize("make_model, oper", [(pvc.Multi2T, "CM"), (pvc.Tandem3T, "MPP"), (pvc.Tandem3T, "VM-21-r")])
def test_run_ey_vectorized_chunks_give_the_same_rows(monkeypatch, make_model, oper):
    m, irr, _ = _synthetic_meteo(ndays=2)
    _populate(m, irr)
    energy_whole, _ = m.run_ey(make_model(), oper)
    whole = m.results.copy()
    monkeypatch.setattr(pvc.EY, "_ROWS_PER_CALL", 7)  # 48 rows --> 7 chunks, the last one short
    energy_chunked, _ = m.run_ey(make_model(), oper)
    np.testing.assert_allclose(energy_chunked, energy_whole, rtol=1e-12)
    # MPP: the flat two-current optimum turns last-ulp differences of the scan into about 1e-6 in Vmp / Imp (Pmp 5e-11)
    rtol = {"Vmp": 1e-5, "Imp": 1e-5} if oper == "MPP" else {}
    for col in whole.columns:
        np.testing.assert_allclose(m.results[col].to_numpy(), whole[col].to_numpy(), rtol=rtol.get(col, 1e-9), err_msg=col)
    assert list(m.results.index) == list(whole.index)


@pytest.mark.parametrize("make_model, oper", [(pvc.Multi2T, "CM"), (pvc.Tandem3T, "CM"), (pvc.Tandem3T, "MPP"), (pvc.Tandem3T, "VM-21-s")])
def test_run_ey_vectorized_zero_output_rows(make_model, oper):
    """Night rows and rows with a non-finite input give exactly zero, as in the per-row solver."""
    m, irr, _ = _synthetic_meteo(ndays=1)
    jsc = irr / 1000 * 20.0
    jsc[12] = np.nan
    eg = np.full(len(irr), 1.7)
    eg[10] = np.nan
    m.add_currents(jsc)
    m.add_currents(irr / 1000 * 18.0)
    m.add_bandgaps(eg)
    m.add_bandgaps(np.full(len(irr), 1.1))
    energy_out, _ = m.run_ey(make_model(), oper)
    res = m.results.to_numpy()
    assert np.isfinite(energy_out) and np.all(np.isfinite(res))
    assert np.all(res[[10, 12]] == 0.0)
    assert np.all(res[irr == 0] == 0.0)
    assert np.all(res[[9, 11, 13], 4] > 0.0)


@pytest.mark.parametrize("make_model, oper", [(pvc.Multi2T, "CM"), (pvc.Tandem3T, "CM"), (pvc.Tandem3T, "MPP"), (pvc.Tandem3T, "VM-21-s"), (pvc.Tandem3T, "VM-21-r")])
def test_run_ey_row_with_negative_jsc_keeps_the_per_row_solver(make_model, oper):
    """An unclipped negative photocurrent has no generating operating point to search for.

    Such a row is solved timestep by timestep in either case, and it must not
    derail the other rows. (Tandem3T.VM used to raise NameError on it.)
    """
    results = {}
    for vectorized in (True, False):
        m, irr, _ = _synthetic_meteo(ndays=1)
        bottom = irr / 1000 * 18.0
        bottom[12] = -0.5  # e.g. from a temperature fit that was not clipped at zero
        m.add_currents(irr / 1000 * 20.0)
        m.add_currents(bottom)
        m.add_bandgaps(np.full(len(irr), 1.7))
        m.add_bandgaps(np.full(len(irr), 1.1))
        m.run_ey(make_model(), oper, multiprocessing=False, vectorized=vectorized)
        results[vectorized] = m.results.to_numpy()
    assert np.all(np.isfinite(results[True]))
    np.testing.assert_array_equal(results[True][12], results[False][12])
    others = np.arange(len(irr)) != 12
    np.testing.assert_allclose(results[True][others, 4], results[False][others, 4], rtol=2e-3)
    assert np.all(results[True][others, 4] >= results[False][others, 4] * (1.0 - 1e-9))


# Unshunted top junction with a negative photocurrent: the 2T stack carries the
# floor current 0.002 A at short circuit, but it has no operating point with power.
_NO_POWER_JSC = [-2.0, 25.433]  # mA/cm^2
_NO_POWER_EG = [1.8, 1.4]


@pytest.mark.parametrize("solver", ["fast", "legacy"])
@pytest.mark.parametrize("make_model", [pvc.Multi2T, pvc.Tandem3T])
def test_calc_yield_async_row_without_power_gives_zero_output(monkeypatch, make_model, solver):
    """A timestep without power reports 0 in every column, as a timestep without light.

    Regression: with the fast solver the row reported the floor current as Isc
    (and as Imp for Multi2T) while Pmp was 0.
    """
    monkeypatch.setattr(pvc.junction, "SOLVER", solver)
    row = pvc.EY._calc_yield_async(np.array([_NO_POWER_JSC]), np.array([_NO_POWER_EG]), np.zeros((1, 2)), pd.Series([25.0]), make_model(), "CM")
    np.testing.assert_array_equal(row.to_numpy(), np.zeros((1, 5)))


@pytest.mark.parametrize("make_model", [pvc.Multi2T, pvc.Tandem3T])
def test_run_ey_row_without_power_gives_zero_output(make_model):
    results = {}
    for vectorized in (True, False):
        m, irr, _ = _synthetic_meteo(ndays=1)
        _populate(m, irr)
        m.jscs[12] = _NO_POWER_JSC  # noon row
        m.bandgaps[12] = _NO_POWER_EG
        m.run_ey(make_model(), "CM", multiprocessing=False, vectorized=vectorized)
        results[vectorized] = m.results.to_numpy()
    np.testing.assert_array_equal(results[True][12], np.zeros(5))
    np.testing.assert_array_equal(results[False][12], np.zeros(5))


def test_run_ey_vectorized_hands_unsolved_rows_to_the_per_row_solver(monkeypatch):
    """A row the *_rows methods return as non-finite is solved per row, not zeroed."""
    solve_2T = pvc.Multi2T.MPP_rows
    solve_3T = pvc.Tandem3T.VM_rows

    def partly_failing_2T(self, *args, **kwargs):
        mpp = solve_2T(self, *args, **kwargs)
        mpp["Pmp"][2] = np.nan  # third lit row of the day
        return mpp

    def partly_failing_3T(self, *args, **kwargs):
        iv3T = solve_3T(self, *args, **kwargs)
        iv3T.Iro[2] = np.nan
        iv3T.Pcalc()  # marks the power of that row with -100
        return iv3T

    for make_model, oper, cls, name, replacement in ((pvc.Multi2T, "CM", pvc.Multi2T, "MPP_rows", partly_failing_2T), (pvc.Tandem3T, "VM-21-s", pvc.Tandem3T, "VM_rows", partly_failing_3T)):
        (_, vec), (_, row), irr = _run_both(make_model, oper)
        lit = np.flatnonzero(irr > 1.0)
        with monkeypatch.context() as patch:
            patch.setattr(cls, name, replacement)
            m, irr, _ = _synthetic_meteo(ndays=1)
            _populate(m, irr)
            m.run_ey(make_model(), oper)
        np.testing.assert_array_equal(m.results.iloc[lit[2]].to_numpy(), row.iloc[lit[2]].to_numpy())
        others = np.arange(len(irr)) != lit[2]
        np.testing.assert_array_equal(m.results.to_numpy()[others], vec.to_numpy()[others])


#################################################
# Test-file generator
#################################################


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
