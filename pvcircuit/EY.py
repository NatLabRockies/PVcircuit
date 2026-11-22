"""
Package to simulate energy yield
"""

import copy
import multiprocessing as mp
from typing import Tuple, Union

import numpy as np  # arrays
import pandas as pd
from loguru import logger
from scipy import constants
from scipy.integrate import trapezoid
from tqdm import tqdm

import pvcircuit as pvc
from pvcircuit import junction

# Below this upper wavelength bound (or above _MAX_UV_START_WAVELENGTH_NM) the
# integrated spectra omit part of the broadband plane-of-array irradiance
# (AM1.5G carries ~0.78 % of its power above 2500 nm, ~5.5 % above 1700 nm, and
# ~1.4 % below 350 nm) -- cell temperature and energy_in would be biased unless
# broadband poa_irradiance is supplied.
_MIN_BROADBAND_WAVELENGTH_NM = 2500.0
_MAX_UV_START_WAVELENGTH_NM = 300.0

# Timesteps solved per call of a *_rows method in run_ey; bounds the size of the work arrays.
_ROWS_PER_CALL = 20000


def VMloss(model: Union["pvc.Tandem3T", "pvc.Multi2T"], oper: str, ncells: int) -> float:
    """
    Calculate the voltage mismatch loss factor for a tandem cell.

    Args:
        model (Union["pvc.Tandem3T", "pvc.Multi2T"]): Either a Tandem3T or Multi2T model.
        oper (str): Operation mode of the tandem cell, e.g., 'VM-21-r' for 3T voltage matched operation, 'MPP' for 4T operation, or 'CM' for 2T operation.
        ncells (int): Number of cells in the string.

    Raises:
        ValueError: If the operation mode format is incorrect or the model type is unknown.

    Returns:
        float: The calculated voltage mismatch loss factor.
    """
    if ncells <= 0:
        raise ValueError(f"VMloss requires ncells >= 1, got {ncells}")

    if isinstance(model, pvc.Multi2T):  # Multi2T or current matched 2-junction tandem
        return 1

    if not isinstance(model, pvc.Tandem3T):
        raise ValueError(f"Unknown model type: {type(model).__name__}")

    # Tandem3T
    tandem_type = oper.split("-")

    if tandem_type[0] in ("MPP", "CM"):
        return 1
    if tandem_type[0] != "VM":
        raise ValueError(f"Unknown 3T operation mode: {oper!r}")

    if len(tandem_type) != 3:
        raise ValueError("3T voltage matched operation must be VM-[bc/tc ratio]-[r/s-type], e.g. VM-21-r")
    vm_ratio = tuple(map(int, tandem_type[1]))

    if tandem_type[2] == "r":
        endloss = max(vm_ratio) - 1
    elif tandem_type[2] == "s":
        endloss = sum(vm_ratio) - 1
    else:
        raise ValueError(f"Unknown VM polarity suffix {tandem_type[2]!r}; must be 'r' or 's'")

    return max(0, 1 - endloss / ncells)


# @lru_cache(maxsize=100)
def VMlist(mmax: int) -> list[str]:
    """
    generate a list of 3T VM configurations + 'MPP'=4T and 'CM'=2T
    mmax < 10 for formating reasons
    """
    if mmax > 9:
        raise ValueError("mmmax must be smaller than 10")

    sVM = ["MPP", "CM"]  # Initialize sVM with predefined elements
    primes = [2, 3, 5]
    for m in range(mmax + 1):
        for n in range(1, m):
            if any(m % p == 0 and n % p == 0 for p in primes):
                continue
            sVM.append(f"VM{m}{n}")
    return sVM


def sandia_T(poa_global: float | np.ndarray | pd.Series, wind_speed: float | np.ndarray | pd.Series, temp_air: float | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    r"""
    Calculate the solar cell temperature using the Sandia model.

    Adapted from the pvlib library to avoid using pandas dataframes.
    Parameters used are those of 'open_rack_cell_polymerback'.

    Args:
        poa_global (float): Plane of array irradiance [W/m^2].
        wind_speed (float): Wind speed [m/s].
        temp_air (float): Ambient air temperature [\degC].

    Returns:
        float: Calculated cell temperature [\degC].
    """
    a = -3.56
    b = -0.075
    deltaT = 3

    E0 = 1000.0  # Reference irradiance

    temp_module = poa_global * np.exp(a + b * wind_speed) + temp_air

    temp_cell = temp_module + (poa_global / E0) * (deltaT)

    return temp_cell


def _time_weights(datetime: pd.DatetimeIndex) -> np.ndarray:
    """
    Per-row integration weights [s] for a (possibly irregular) time axis.

    Row i gets half of the gap to its predecessor plus half of the gap to its
    successor (end rows: half of their single gap), so that
    ``sum(y * w) == trapezoid(y, t)`` for the full timeline. Because the
    weights are stored per row they survive row filtering: dropping rows
    (night, NaN, APE filters) simply removes their contribution instead of
    letting the trapezoid rule bridge the gap.
    """
    n = len(datetime)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.zeros(1)
    seconds = (datetime - datetime[0]).total_seconds().to_numpy(dtype=np.float64)
    gaps = np.diff(seconds)
    w = np.empty(n, dtype=np.float64)
    w[0] = gaps[0] / 2
    w[-1] = gaps[-1] / 2
    w[1:-1] = (gaps[:-1] + gaps[1:]) / 2
    return w


def _calc_yield_async(Jscs: np.ndarray, Egs: np.ndarray, sigmas: np.ndarray, TempCell: pd.Series, model: Union["pvc.Multi2T", "pvc.Tandem3T"], oper: str) -> pd.DataFrame:
    """Evaluate IV parameters for one chunk of timesteps on a single device copy.

    The same model instance is reused for every row: each iteration overwrites
    all state that varies between rows (Eg, sigma, Jext, TC), so no state leaks
    from one timestep to the next.

    Rows without photocurrent (all Jsc == 0, e.g. night) or with non-finite
    current, bandgap, sigma, or cell temperature produce zero output
    (Pmp = Isc = Imp = 0, Voc = Vmp = 0). Multi2T.MPP() returns NaN at zero
    photocurrent, which would otherwise poison the time integral of the year.
    """

    columns: list[str] = ["Voc", "Isc", "Vmp", "Imp", "Pmp"]
    IV_params = pd.DataFrame(np.zeros((len(Jscs), len(columns))), columns=columns)

    for i in range(len(Jscs)):
        row_jsc = np.asarray(Jscs[i], dtype=np.float64)
        row_eg = np.asarray(Egs[i], dtype=np.float64)
        row_sigma = np.asarray(sigmas[i], dtype=np.float64)
        row_temp = float(TempCell.iloc[i])
        # Same threshold as Multi2T.MPP (1e-6 A/cm^2 = 1e-3 mA/cm^2), which
        # returns NaN below it.
        if (
            not np.all(np.isfinite(row_jsc))
            or not np.all(np.isfinite(row_eg))
            or not np.all(np.isfinite(row_sigma))
            or not np.isfinite(row_temp)
            or np.max(row_jsc) <= 1e-3
        ):
            # zero output; the DataFrame is pre-filled with zeros
            continue

        if isinstance(model, pvc.Multi2T):  # Multi2T or current matched 2-junction tandem
            for ijunc in range(model.njuncs):
                # Jscs is stored in mA/cm^2 (Meteo.add_currents); junction.Jext
                # expects A/cm^2, so divide by 1000.
                model.j[ijunc].set(Eg=row_eg[ijunc], sigma=row_sigma[ijunc], Jext=row_jsc[ijunc] / 1e3, TC=row_temp)

            mpp_dict = model.MPP()
            # Pmax = mpp_dict["Pmp"]
            for col in columns:
                IV_params.loc[i, col] = mpp_dict[col]

        elif isinstance(model, pvc.Tandem3T):  # Tandem3T
            tandem_type = oper.split("-")

            # Jscs in mA/cm^2 -> A/cm^2 for junction.Jext, same conversion as above.
            model.top.set(Eg=row_eg[0], sigma=row_sigma[0], Jext=row_jsc[0] / 1e3, TC=row_temp)
            model.bot.set(Eg=row_eg[1], sigma=row_sigma[1], Jext=row_jsc[1] / 1e3, TC=row_temp)
            if tandem_type[0] == "MPP":
                tempRz = model.Rz
                model.set(Rz=0)
                iv3T = model.MPP()
                model.set(Rz=tempRz)
            elif tandem_type[0] == "CM":
                ln, iv3T = model.CM()
            elif tandem_type[0] == "VM":
                if len(tandem_type) != 3 or tandem_type[2] not in ("r", "s"):
                    raise ValueError("3T voltage matched operation must be VM-[bc/tc ratio]-[r/s-type], e.g. VM-21-r")
                # pvcircuit convention: equal pn is r-type (junctions reversed
                # against each other), opposite pn is s-type (series connected)
                model.bot.pn = model.top.pn if tandem_type[2] == "r" else -1 * model.top.pn
                ln, iv3T = model.VM(*map(int, tandem_type[1]))
            else:
                raise ValueError(f"Unknown 3T operation mode: {oper!r}")
            assert isinstance(iv3T, pvc.iv3T.IV3T)
            # Load-terminal quantities: VA - VB and min(|IA|, |IB|) equal the
            # tandem Vmp/Imp (and Voc below) for CM operation; for MPP/VM they
            # are the terminal values of the load configuration. Pmp (Ptot) is
            # the total device output in every mode.
            IV_params.loc[i, "Vmp"] = iv3T.VA - iv3T.VB
            IV_params.loc[i, "Imp"] = min(abs(iv3T.IA), abs(iv3T.IB))
            IV_params.loc[i, "Pmp"] = iv3T.Ptot

            iv3T = model.Voc3()
            IV_params.loc[i, "Voc"] = iv3T.VA - iv3T.VB

            if tandem_type[0] == "CM" and model.top.pn * model.bot.pn == -1:
                # CM leaves the z contact floating, so the short circuit is the
                # 2T one (Vtr = 0, Izo = 0). Isc3 shorts all three terminals and
                # lets z carry the mismatch current.
                IV_params.loc[i, "Isc"] = pvc.Multi2T.from_3T(model).Isc()
            else:
                iv3T = model.Isc3()
                IV_params.loc[i, "Isc"] = min(abs(iv3T.IA), abs(iv3T.IB))

        else:
            raise ValueError(f"Unknown model type: {type(model).__name__}")

    # Any remaining non-finite value (e.g. a solver failure on a single row)
    # must not poison the yield integral.
    if not np.all(np.isfinite(IV_params.to_numpy())):
        nbad = int((~np.isfinite(IV_params.to_numpy())).any(axis=1).sum())
        logger.warning("_calc_yield_async: {} of {} timesteps returned non-finite IV parameters; set to 0", nbad, len(IV_params))
        # No operating point (0 in every column, as without light): a non-finite Pmp, or the placeholder point
        # Tandem3T.CM() / VM() return when no point has positive power (Ptot 0.0, VA / VB / IA / IB nan).
        Pmp = IV_params["Pmp"].to_numpy()
        no_point = ~np.isfinite(Pmp) | (~(Pmp > 0.0) & ~np.isfinite(IV_params[["Vmp", "Imp"]].to_numpy()).all(axis=1))
        IV_params.loc[no_point, :] = 0.0
        IV_params = IV_params.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    return IV_params  # Pmax in [W]


def _solves_all_rows(model: Union["pvc.Multi2T", "pvc.Tandem3T"], oper: str) -> bool:
    """Whether run_ey can solve all timesteps at once with the *_rows methods of ``model``.

    True for every Multi2T, and for a Tandem3T in "MPP" and "VM-xy-r/s"
    operation and in "CM" operation with series-connected junctions (s-type,
    opposite pn; with the z contact floating that is the 2T stack
    Multi2T.from_3T(model)). Everything else is solved timestep by timestep.
    """
    if isinstance(model, pvc.Multi2T):
        return True
    if not isinstance(model, pvc.Tandem3T) or model.top.beta > 0.0 or model.top.notdiode() or model.bot.notdiode():
        return False
    tandem_type = oper.split("-")
    if tandem_type[0] == "CM":
        return model.top.pn * model.bot.pn == -1
    if tandem_type[0] == "MPP":
        return True
    if tandem_type[0] == "VM":
        return len(tandem_type) == 3 and tandem_type[2] in ("r", "s") and tandem_type[1].isdigit() and len(tandem_type[1]) == 2 and "0" not in tandem_type[1]
    return False


def _solve_rows(model: Union["pvc.Multi2T", "pvc.Tandem3T"], oper: str, TC: np.ndarray, Eg: np.ndarray, sigma: np.ndarray, Jext: np.ndarray) -> np.ndarray:
    """IV parameters (Voc, Isc, Vmp, Imp, Pmp) of all rows at once: the *_rows twin of the loop body of _calc_yield_async.

    Jext in A/cm^2. Returns a (rows, 5) array with np.nan in every row that
    has no solution. ``model`` and ``oper`` must satisfy _solves_all_rows.
    """
    if isinstance(model, pvc.Multi2T):
        mpp = model.MPP_rows(TC, Eg, sigma, Jext)
        return np.column_stack([mpp[col] for col in ("Voc", "Isc", "Vmp", "Imp", "Pmp")])

    tandem_type = oper.split("-")
    if tandem_type[0] == "CM":
        mpp = pvc.Multi2T.from_3T(model).MPP_rows(TC, Eg, sigma, Jext)
        # the load voltage VA - VB of a Tandem3T follows the polarity of the device
        vsign = -float(model.top.pn)
        return np.column_stack([vsign * mpp["Voc"], mpp["Isc"], vsign * mpp["Vmp"], mpp["Imp"], mpp["Pmp"]])

    dev = model.copy()
    if tandem_type[0] == "MPP":
        dev4T = model.copy()
        dev4T.set(Rz=0)
        iv3T = dev4T.MPP_rows(TC, Eg, sigma, Jext)
    else:
        # pvcircuit convention: equal pn is r-type, opposite pn is s-type
        dev.bot.pn = dev.top.pn if tandem_type[2] == "r" else -1 * dev.top.pn
        iv3T = dev.VM_rows(*map(int, tandem_type[1]), TC, Eg, sigma, Jext)
    voc = dev.Voc3_rows(TC, Eg, sigma, Jext)
    isc = dev.Isc3_rows(TC, Eg, sigma, Jext)
    values = np.column_stack([voc.VA - voc.VB, np.minimum(np.abs(isc.IA), np.abs(isc.IB)), iv3T.VA - iv3T.VB, np.minimum(np.abs(iv3T.IA), np.abs(iv3T.IB)), iv3T.Ptot])
    # IV3T.Pcalc marks an undefined power with -100 instead of np.nan
    solved = np.isfinite(iv3T.IA) & np.isfinite(iv3T.IB) & np.isfinite(iv3T.VA) & np.isfinite(iv3T.VB)
    values[~solved] = np.nan
    return values


def _calc_yield_rows(Jscs: np.ndarray, Egs: np.ndarray, sigmas: np.ndarray, TempCell: pd.Series, model: Union["pvc.Multi2T", "pvc.Tandem3T"], oper: str) -> pd.DataFrame:
    """Evaluate IV parameters for one chunk of timesteps, all rows at once.

    Twin of _calc_yield_async for the devices and modes of _solves_all_rows,
    with the same zero-output rule for rows without photocurrent or with
    non-finite inputs. A row the *_rows methods do not solve, and a row with
    a negative photocurrent, is handed to _calc_yield_async, so the result is
    never worse than timestep by timestep.
    """
    columns: list[str] = ["Voc", "Isc", "Vmp", "Imp", "Pmp"]
    Jscs = np.asarray(Jscs, dtype=np.float64)
    Egs = np.asarray(Egs, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    temp = TempCell.to_numpy(dtype=np.float64)
    out = np.zeros((len(Jscs), len(columns)))

    # Same threshold as Multi2T.MPP (1e-6 A/cm^2 = 1e-3 mA/cm^2).
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(Jscs).all(axis=1) & np.isfinite(Egs).all(axis=1) & np.isfinite(sigmas).all(axis=1) & np.isfinite(temp) & (np.max(Jscs, axis=1) > 1e-3)
        # A negative photocurrent has no generating operating point for the
        # *_rows methods to search; such a row keeps the timestep-by-timestep solver.
        negative = valid & (Jscs < 0.0).any(axis=1)
    rows = np.flatnonzero(valid & ~negative)
    failed = np.flatnonzero(negative)
    if rows.size:
        # Jscs is stored in mA/cm^2 (Meteo.add_currents); the devices expect A/cm^2.
        values = _solve_rows(model, oper, temp[rows], Egs[rows], sigmas[rows], Jscs[rows] / 1e3)
        out[rows] = values
        failed = np.concatenate([failed, rows[~np.isfinite(values).all(axis=1)]])
    if failed.size:
        logger.warning("_calc_yield_rows: {} of {} timesteps not solved all at once; solving them timestep by timestep", failed.size, len(Jscs))
        out[failed] = _calc_yield_async(Jscs[failed], Egs[failed], sigmas[failed], TempCell.iloc[failed], copy.deepcopy(model), oper).to_numpy()

    return pd.DataFrame(out, columns=columns)  # Pmax in [W]


def _set_solver(solver: str) -> None:
    """Pool initializer: worker processes follow the solver setting of the parent."""
    junction.SOLVER = solver


class Meteo:
    """
    Handles meteorological environmental data and spectral information for energy yield simulations.

    NOTE: Construct 'Meteo' on a complete, strictly increasing timeline (retain
    night rows and pass missing measurements as NaN rows rather than dropping
    them beforehand, otherwise trapezoid weights bridge the missing intervals).
    Use 'filter_custom', 'filter_ape', or 'filter_spectra' *after* construction
    to select rows while preserving their original integration weights 'dt'.

    Row-aligned attributes: 'datetime', 'temp', 'wind', 'spectra', 'irradiance',
    'cell_temp', 'dt' (integration weight per row [s]), 'average_photon_energy',
    'jscs', 'bandgaps', 'sigmas' and, after 'run_ey', 'results'.

    Time integration ('energy_in' and energy yield from 'run_ey') integrates over
    the covered period in [kWh/m^2] (equal to [kWh/m^2/yr] for a one-year record).

    Non-finite spectral entries are replaced by 0 (a partially missing spectrum
    still contributes its remaining bands to the irradiance). Timestamps with
    non-finite ambient temperature or wind are retained with a non-finite cell
    temperature; 'run_ey' assigns zero output to those rows while keeping their
    irradiance in 'energy_in'. Feed the *filled* spectra ('Meteo.spectra',
    'Meteo.wavelength') to 'EQE.add_spectra' so that the Jsc time series stays
    aligned with 'cell_temp'.
    """

    def __init__(
        self,
        wavelength: np.ndarray,
        spectra: pd.DataFrame,
        ambient_temperature: pd.Series,
        wind: pd.Series,
        datetime: pd.DatetimeIndex,
        poa_irradiance: pd.Series | np.ndarray | None = None,
    ) -> None:
        wavelength = np.asarray(wavelength, dtype=np.float64).flatten()
        datetime = pd.DatetimeIndex(datetime)
        nrows = len(datetime)

        if nrows >= 2:
            dt_secs = (datetime - datetime[0]).total_seconds().to_numpy(dtype=np.float64)
            gaps = np.diff(dt_secs)
            if np.any(gaps <= 0.0):
                raise ValueError("Meteo: datetime index must be strictly increasing")
            if nrows >= 3:
                med_gap = float(np.median(gaps))
                max_gap = float(gaps.max())
                if med_gap > 0.0 and max_gap > 2.0 * med_gap:
                    logger.warning(
                        "Meteo: timeline contains gaps up to {:.2f} h (median step {:.2f} h); trapezoid weights bridge across dropped rows. Pass a complete timeline (using NaN for missing rows) or filter after Meteo() construction.",
                        max_gap / 3600.0,
                        med_gap / 3600.0,
                    )
        elif nrows == 1:
            logger.warning("Meteo: initialized with a single timestamp; integration weight dt is 0 s (energy_in and yield will be 0).")

        def _align_series(val, label: str) -> pd.Series:
            if isinstance(val, pd.Series) and isinstance(val.index, pd.DatetimeIndex) and len(val) == nrows and not val.index.equals(datetime):
                if len(val.index.unique()) == nrows and set(val.index) == set(datetime):
                    logger.warning("Meteo: realigning {} on its DatetimeIndex to match datetime", label)
                    val = val.reindex(datetime)
                else:
                    logger.warning("Meteo: {} DatetimeIndex differs from datetime; aligning positionally", label)
            return pd.Series(np.asarray(val, dtype=np.float64))

        if isinstance(spectra, pd.DataFrame) and isinstance(spectra.index, pd.DatetimeIndex) and spectra.shape[0] == nrows and not spectra.index.equals(datetime):
            if len(spectra.index.unique()) == nrows and set(spectra.index) == set(datetime):
                logger.warning("Meteo: realigning spectra on its DatetimeIndex to match datetime")
                spectra = spectra.reindex(datetime)
            else:
                logger.warning("Meteo: spectra DatetimeIndex differs from datetime; aligning positionally")
        spectra = pd.DataFrame(spectra).copy()
        ambient_temperature = _align_series(ambient_temperature, "ambient_temperature")
        wind = _align_series(wind, "wind")

        for name, length in (("spectra", spectra.shape[0]), ("ambient_temperature", len(ambient_temperature)), ("wind", len(wind))):
            if length != nrows:
                raise ValueError(f"Meteo: {name} has {length} rows but datetime has {nrows} entries; all inputs must be row-aligned")
        if len(wavelength) < 2:
            raise ValueError("Meteo: wavelength must have at least 2 entries")
        if spectra.shape[1] != len(wavelength):
            raise ValueError(f"Meteo: spectra has {spectra.shape[1]} columns but wavelength has {len(wavelength)} entries; spectra must be (n_times, n_wavelengths)")

        if not np.all(np.diff(wavelength) > 0.0):
            if np.any(np.diff(np.sort(wavelength)) == 0.0):
                raise ValueError("Meteo: wavelength contains duplicate entries")
            logger.warning("Meteo: sorting non-ascending wavelength axis")
            wl_order = np.argsort(wavelength, kind="stable")
            wavelength = wavelength[wl_order]
            spectra = spectra.iloc[:, wl_order]

        # Retain the complete timeline. Missing spectral bands carry no power,
        # and rows lacking weather inputs are made explicit zero-output rows by
        # run_ey rather than being removed and bridged during time integration.
        nonfinite_spectra = ~np.isfinite(spectra.to_numpy(dtype=np.float64))
        n_nonfinite_spectra_rows = int(nonfinite_spectra.any(axis=1).sum())
        if n_nonfinite_spectra_rows > 0:
            logger.warning(
                "Meteo: {} of {} spectra rows contain non-finite values; they are filled with 0 (partial irradiance still counts). Use Meteo.spectra for EQE.add_spectra so Jsc stays finite.",
                n_nonfinite_spectra_rows,
                nrows,
            )
        spectra = spectra.mask(nonfinite_spectra, 0.0)
        invalid_weather = ~np.isfinite(ambient_temperature.to_numpy()) | ~np.isfinite(wind.to_numpy())
        if invalid_weather.any():
            logger.warning(
                "Meteo: retaining {} rows with non-finite temperature/wind; run_ey will assign zero output to those timestamps",
                int(invalid_weather.sum()),
            )
        # `spectra` keeps its own row index (whatever the caller used); the
        # datetime is kept separately in self.datetime.
        self.temp = pd.Series(ambient_temperature.to_numpy(), index=datetime)  # Ambient temperature in degrees Celsius
        self.wind = pd.Series(wind.to_numpy(), index=datetime)  # Wind speed in meters per second
        self.datetime = datetime

        self.wavelength = wavelength
        self.spectra = spectra

        if poa_irradiance is None:
            if wavelength.max() <= _MIN_BROADBAND_WAVELENGTH_NM or wavelength.min() > _MAX_UV_START_WAVELENGTH_NM:
                logger.warning(
                    "Meteo: spectra cover [{:.0f}, {:.0f}] nm rather than full broadband (~280-4000 nm). Irradiance (cell temperature) and energy_in are integrated over the given range only unless poa_irradiance is supplied.",
                    wavelength.min(),
                    wavelength.max(),
                )
            irr_vals = trapezoid(y=self.spectra.to_numpy(), x=self.wavelength)
        else:
            poa_s = _align_series(poa_irradiance, "poa_irradiance")
            if len(poa_s) != nrows:
                raise ValueError(f"Meteo: poa_irradiance has {len(poa_s)} entries, expected {nrows}")
            irr_vals = poa_s.to_numpy(dtype=np.float64)

        fin_irr = irr_vals[np.isfinite(irr_vals)]
        if fin_irr.size and (np.any(fin_irr < -1.0) or np.any(fin_irr > 1600.0)):
            logger.warning(
                "Meteo: integrated irradiance [{:.1f}, {:.1f}] W/m^2 is outside expected solar bounds [0, 1600] W/m^2; check spectral units (expected W/m^2/nm).",
                float(fin_irr.min()),
                float(fin_irr.max()),
            )

        self.irradiance = pd.Series(irr_vals, index=self.datetime)  # Optical power of each spectrum [W/m^2]
        cell_temp = np.asarray(
            sandia_T(self.irradiance.to_numpy(), self.wind.to_numpy(), self.temp.to_numpy()),
            dtype=np.float64,
        )
        cell_temp[invalid_weather] = np.nan
        self.cell_temp: pd.Series = pd.Series(cell_temp, index=self.datetime)  # Cell temperature calculation
        # Per-row integration weights [s] (see _time_weights). Uses
        # `(datetime - datetime[0]).total_seconds()`, which is pandas-native and
        # resolution-agnostic (pandas 2.x [ns] and 3.x [us]).
        self.dt: pd.Series = pd.Series(_time_weights(self.datetime), index=self.datetime)
        self.energy_in = 0.0
        self._recompute_energy_in()  # Energy input [kWh/m^2/yr]

        self.average_photon_energy = None  # Will be calculated when running calc_ape
        self.jscs = None  # Short-circuit currents
        self.bandgaps = None  # Bandgap energies
        self.sigmas = None  # Sigma values
        self.results: pd.DataFrame | None = None

    def _add_array(self, array: np.ndarray | pd.Series | pd.DataFrame, attribute_name: str) -> None:
        """
        Helper function to add data arrays (e.g., jsc, bandgap, sigma) to the instance.

        Args:
            array (np.ndarray): Array to add.
            attribute_name (str): Name of the attribute to which the array will be added.
        """
        method_label = "add_" + {"jscs": "currents", "bandgaps": "bandgaps", "sigmas": "sigmas"}.get(attribute_name, attribute_name)
        nrows = self.cell_temp.shape[0]
        if isinstance(array, (pd.Series, pd.DataFrame)) and isinstance(array.index, pd.DatetimeIndex) and len(array) == nrows and not array.index.equals(self.datetime):
            if len(array.index.unique()) == nrows and set(array.index) == set(self.datetime):
                logger.warning("Meteo.{}: realigning input on its DatetimeIndex to match datetime", method_label)
                array = array.reindex(self.datetime)
            else:
                logger.warning("Meteo.{}: input DatetimeIndex differs from datetime; aligning positionally", method_label)

        arr = np.asarray(array, dtype=np.float64)
        # Ensure the array is columnar and matches the number of rows in cell_temp
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.shape[1] == nrows and arr.shape[0] == 1:
            arr = arr.T
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1] == nrows and nrows > 1:
            logger.warning(
                "Meteo.{}: square ({}, {}) input interpreted as (n_times, n_junctions); note that EQE.Jint() returns (n_junctions, n_times) and requires .T",
                method_label,
                nrows,
                nrows,
            )

        if arr.shape[0] != nrows:
            raise ValueError(f"Shape of data {arr.shape} does not match cell_temp rows {nrows}")

        if not np.all(np.isfinite(arr)):
            logger.warning(
                "Meteo.{}: {} of {} rows contain non-finite values; those timesteps will yield zero output in run_ey",
                method_label,
                int((~np.isfinite(arr)).any(axis=1).sum()),
                arr.shape[0],
            )

        current_attr = getattr(self, attribute_name)
        if current_attr is None:
            setattr(self, attribute_name, arr)
        else:
            setattr(self, attribute_name, np.concatenate((current_attr, arr), axis=1))

    def add_currents(self, jsc: np.ndarray) -> None:
        """
        Add Jsc array to the instance, one call (or one column) per junction.

        Jsc values are stored in self.jscs in **mA/cm^2** (the native unit
        of JintMD).  They are converted to A/cm^2 inside
        _calc_yield_async before being assigned to
        Junction.Jext.

        Args:
            jsc (np.ndarray): Short-circuit current values to add [mA/cm^2], one
                value per timestep (shape ``(n_times,)``, ``(n_times, 1)`` or
                ``(1, n_times)``). For temperature-dependent EQE use
                ``EQET.get_current_for_temperature(cell_temp)``; do not loop over
                the rows of ``EQET.Jint()`` (those are the *measured* temperatures).
        """
        self._add_array(jsc, "jscs")

    def add_bandgaps(self, bandgap: np.ndarray) -> None:
        """
        Add bandgap array to the instance, one call (or one column) per junction.

        Args:
            bandgap (np.ndarray): Bandgap energy values to add [eV], one per timestep.
        """
        self._add_array(bandgap, "bandgaps")

    def add_sigmas(self, sigma: np.ndarray) -> None:
        """
        Add sigma array for bandgap tail states to the instance, one call (or one column) per junction.

        Args:
            sigma (np.ndarray): Sigma values to add [eV], one per timestep.
        """
        self._add_array(sigma, "sigmas")

    @staticmethod
    def _model_njuncs(model: Union["pvc.Multi2T", "pvc.Tandem3T"]) -> int:
        if isinstance(model, pvc.Multi2T):
            return int(model.njuncs)
        if isinstance(model, pvc.Tandem3T):
            return 2
        raise ValueError(f"Unknown model type: {type(model).__name__}")

    def run_ey(self, model: Union["pvc.Multi2T", "pvc.Tandem3T"], oper: str, multiprocessing: bool = True, vectorized: Union[bool, None] = None) -> tuple[float, float]:
        """
        Calculate the energy yield and efficiency based on the provided model and operation mode.

        Args:
            model (Union["pvc.Multi2T", "pvc.Tandem3T"]): Either a Multi2T or Tandem3T model.
            oper (str): Operation mode, e.g., 'MPP', 'CM', 'VM-21-r', 'VM-21-s'.
            multiprocessing (bool, optional): Whether to use multiprocessing for the
                timestep-by-timestep solver. Defaults to True.
            vectorized (bool, optional): Solve all timesteps at once with the *_rows
                methods of the device (Multi2T.MPP_rows, Tandem3T.MPP_rows / VM_rows);
                ``multiprocessing`` is not used then. False solves timestep by
                timestep. Also with True, timestep by timestep are solved: the whole
                run of a Tandem3T that is r-type in 'CM' operation, has top.beta > 0
                or a resistor-only junction, or has a 'VM' label other than three
                parts with a two-digit ratio of non-zero digits and an 'r' or 's'
                suffix, such as 'VM-21-r' (with the process pool when
                ``multiprocessing`` is True); the rows with a negative photocurrent
                and the rows the *_rows methods return unsolved (in the calling
                process, whatever ``multiprocessing`` is). Defaults to None: True
                unless pvcircuit.junction.SOLVER is "legacy".

        Raises:
            ValueError: If data array sizes are inconsistent with cell temperature, or if
                the number of Jsc/bandgap/sigma columns does not match the number of
                junctions of ``model``.

        Returns:
            Tuple[float, float]: Integrated energy yield [kWh/m^2] over the covered
                Meteo period (i.e. [kWh/m^2/yr] for a one-year record) and energy
                harvesting efficiency 'EnergyOut / energy_in' (note: zero-output
                rows with non-finite weather/Jsc/Eg/sigma keep their irradiance in
                'energy_in', lowering efficiency by their irradiance share).
        """
        if self.jscs is None or self.bandgaps is None:
            raise ValueError("run_ey: add_currents() and add_bandgaps() must be called before run_ey()")

        # Use a local zero array when sigmas were not added so self.sigmas stays None
        # for any later add_sigmas() calls on this Meteo instance.
        sigmas_arr = np.zeros_like(self.bandgaps) if self.sigmas is None else self.sigmas

        assert self.jscs is not None
        assert self.bandgaps is not None

        # Ensure all data arrays have consistent shapes
        njuncs = self._model_njuncs(model)
        for name, attr in (("jscs", self.jscs), ("bandgaps", self.bandgaps), ("sigmas", sigmas_arr)):
            if attr.shape[0] != self.cell_temp.shape[0]:
                raise ValueError(f"Inconsistent array size: {name} has {attr.shape[0]} rows, expected {self.cell_temp.shape[0]}")
            if attr.shape[1] != njuncs:
                raise ValueError(f"run_ey: {name} has {attr.shape[1]} columns but {model.name!r} ({type(model).__name__}) has {njuncs} junctions; call add_{'currents' if name == 'jscs' else name}() exactly once per junction")

        if not np.all(np.isfinite(self.jscs)):
            logger.warning("run_ey: {} timesteps have non-finite Jsc and will yield zero output", int((~np.isfinite(self.jscs)).any(axis=1).sum()))

        # Determine chunk sizes for multiprocessing
        max_chunk_size = 200
        cpu_count = mp.cpu_count()
        chunk_ids = np.arange(len(self.jscs))
        chunk_size = min(len(chunk_ids) // cpu_count + 1, max_chunk_size)

        # Create chunks of data for parallel processing. One deep copy of the
        # device per chunk is enough: _calc_yield_async overwrites all
        # row-varying state (Eg, sigma, Jext, TC) on every iteration.
        chunks = [chunk_ids[i : i + chunk_size] for i in range(0, len(chunk_ids), chunk_size)]

        if vectorized is None:
            vectorized = not junction._legacy_solver()
        all_rows = bool(vectorized) and _solves_all_rows(model, oper)

        with tqdm(total=len(self.datetime), leave=True) as pbar:

            def update_tqdm(*args):
                """Callback function to update the progress bar."""
                pbar.update(len(args[0]))
                pbar.refresh()

            if all_rows:
                pbar.set_description(f"Running {model.name} in mode {oper} vectorized")

                result_dfs = []
                for start in range(0, len(chunk_ids), _ROWS_PER_CALL):
                    chunk = chunk_ids[start : start + _ROWS_PER_CALL]
                    result_dfs.append(_calc_yield_rows(self.jscs[chunk], self.bandgaps[chunk], sigmas_arr[chunk], self.cell_temp.iloc[chunk], model, oper))
                    pbar.update(len(chunk))
                    pbar.refresh()

                results = pd.concat(result_dfs, ignore_index=True)

            elif multiprocessing:
                pbar.set_description(f"Running {model.name} in mode {oper} with {cpu_count} processes")

                with mp.Pool(cpu_count, initializer=_set_solver, initargs=(junction.SOLVER,)) as pool:
                    # Assign tasks to multiprocessing pool
                    jobs = [
                        pool.apply_async(_calc_yield_async, args=(self.jscs[chunk], self.bandgaps[chunk], sigmas_arr[chunk], self.cell_temp.iloc[chunk], copy.deepcopy(model), oper), callback=update_tqdm)
                        for chunk in chunks
                    ]
                    # Collect and combine dataframe results
                    results = pd.concat([job.get() for job in jobs], ignore_index=True)

            else:
                pbar.set_description(f"Running {model.name} in mode {oper} without multiprocessing")

                # For sequential processing, collect dataframes in a list then combine
                result_dfs = []
                for i, chunk in enumerate(chunks):
                    # Process each chunk sequentially
                    chunk_result = _calc_yield_async(self.jscs[chunk], self.bandgaps[chunk], sigmas_arr[chunk], self.cell_temp.iloc[chunk], copy.deepcopy(model), oper)
                    result_dfs.append(chunk_result)
                    pbar.update(len(chunk))
                    pbar.refresh()

                # Combine all dataframes
                results = pd.concat(result_dfs, ignore_index=True)

        self.results = results
        self.results.index = self.datetime

        power = results["Pmp"].to_numpy()  # output power [W]

        # Normalise by device.totalarea (= max junction totalarea). This is
        # the project-wide reference area for power normalisation: it matches
        # how Multi2T.Rs2T is consumed in V2T and what Multi2T.efficiency()
        # and Tandem3T.efficiency() use, so kWh/m^2/yr corresponds to the
        # device footprint a system installer would see.
        power_density = power / model.totalarea  # W --> W/cm^2

        # Weighted sum with the per-row time weights (== trapezoid on the
        # unfiltered timeline; robust against filtered/irregular rows).
        EnergyOut = float(np.sum(power_density * self.dt.to_numpy()))  # [Ws/cm^2] over period

        # Unit conversion breakdown for the factor /3.6e3 /1e3 *1e4:
        #   /3600  : seconds -> hours      (Ws -> Wh)
        #   /1000  : W -> kW               (Wh -> kWh)
        #   *1e4   : cm^-2 -> m^-2         (per-cm^2 -> per-m^2, since 1 m^2 = 1e4 cm^2)
        EnergyOut = EnergyOut / 3.6e3 / 1e3 * 1e4  # [Ws/cm^2] --> [kWh/m^2] over period

        # Calculate energy harvesting efficiency
        EYeff = EnergyOut / self.energy_in if self.energy_in > 0 else np.nan
        return EnergyOut, EYeff

    def calc_ape(self, wavelength_range: tuple[float, float] | None = None) -> None:
        """Calculate the average photon energy (APE) [eV] of the spectra.

        Note: APE depends strongly on the spectral wavelength bounds (e.g. AM1.5G
        is 1.450 eV over 280-4000 nm vs 1.602 eV over 300-1700 nm). Pass
        'wavelength_range=(wl_min, wl_max)' [nm] to evaluate APE over a specific
        band without mutating 'self.wavelength' or 'self.spectra'. The result is
        stored in 'self.average_photon_energy'.
        """
        self.average_photon_energy = self._compute_ape(wavelength_range)

    def _compute_ape(self, wavelength_range: tuple[float, float] | None = None) -> np.ndarray:
        """[eV] average photon energy of every row over 'wavelength_range' [nm] (default: all wavelengths)."""
        wl = self.wavelength
        spec_vals = self.spectra.to_numpy(dtype=np.float64)
        if wavelength_range is not None:
            lo, hi = float(min(wavelength_range)), float(max(wavelength_range))
            keep = (wl >= lo) & (wl <= hi)
            if np.count_nonzero(keep) < 2:
                raise ValueError(f"calc_ape: wavelength_range {wavelength_range} leaves fewer than 2 spectral points")
            wl = wl[keep]
            spec_vals = spec_vals[:, keep]

        phi = spec_vals * (wl[np.newaxis, :] * 1e-9) / constants.h / constants.c
        mask = np.all(phi == 0.0, axis=1)
        phi[mask, :] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            return trapezoid(x=wl, y=spec_vals) / constants.e / trapezoid(x=wl, y=phi)

    def _recompute_energy_in(self) -> None:
        """Recompute the integrated energy input [kWh/m^2] over the covered period from current rows and dt."""
        if len(self.datetime) == 0:
            self.energy_in = 0.0
            return
        self.energy_in = float(np.nansum(self.irradiance.to_numpy() * self.dt.to_numpy())) / 3600 / 1000  # [Ws/m^2] -> [kWh/m^2]

    def _apply_row_mask(self, mask: np.ndarray) -> "Meteo":
        """
        Return a deep copy with a boolean row mask applied to every
        row-aligned attribute (meteorological series, time weights, any added
        jsc/bandgap/sigma arrays and results), and energy_in recomputed.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self.datetime),):
            raise ValueError(f"row mask has shape {mask.shape}, expected ({len(self.datetime)},)")

        self_copy = copy.deepcopy(self)

        self_copy.datetime = self_copy.datetime[mask]
        self_copy.temp = self_copy.temp[mask]
        self_copy.wind = self_copy.wind[mask]
        self_copy.spectra = self_copy.spectra[mask]
        self_copy.irradiance = self_copy.irradiance[mask]
        self_copy.cell_temp = self_copy.cell_temp[mask]
        self_copy.dt = self_copy.dt[mask]
        if self_copy.average_photon_energy is not None:
            self_copy.average_photon_energy = self_copy.average_photon_energy[mask]
        for attr_name in ("jscs", "bandgaps", "sigmas"):
            attr = getattr(self_copy, attr_name)
            if attr is not None:
                setattr(self_copy, attr_name, attr[mask])
        if self_copy.results is not None:
            self_copy.results = self_copy.results[mask]

        # energy_in was integrated over the unfiltered rows; recompute so
        # run_ey efficiency on the filtered copy is normalised correctly.
        self_copy._recompute_energy_in()
        return self_copy

    def filter_ape(self, min_ape: float = 0, max_ape: float = 10, wavelength_range: tuple[float, float] | None = None) -> "Meteo":
        """
        Filter the average photon energy (APE) [eV] within specified bounds.

        Note: Literature APE bounds depend on the spectroradiometer wavelength
        window (see 'calc_ape'); pass 'wavelength_range=(wl_min, wl_max)' [nm]
        when applying bounds defined over a restricted spectral range. That APE
        is computed for this filter only; the stored 'average_photon_energy'
        (used without 'wavelength_range', calculated over all wavelengths when
        missing) is left unchanged.
        """
        if wavelength_range is not None:
            ape = self._compute_ape(wavelength_range)
        else:
            if self.average_photon_energy is None:
                self.calc_ape()
            ape = self.average_photon_energy
        assert ape is not None

        # NaN APE rows (all-zero spectra) compare False and are dropped
        ape_mask = (ape > min_ape) & (ape < max_ape)
        return self._apply_row_mask(np.asarray(ape_mask))

    def filter_spectra(self, min_spectra: float = -1e-6, max_spectra: float = 10) -> "Meteo":
        """
        Filter the spectral data within specified bounds [W/m^2/nm].

        Defaults 'min_spectra=-1e-6' to tolerate tiny floating-point baseline noise
        on dawn/dusk rows; pass 'min_spectra=0.0' for strict non-negativity.
        """
        spectra_mask = (self.spectra >= min_spectra).all(axis=1) & (self.spectra < max_spectra).all(axis=1)
        return self._apply_row_mask(np.asarray(spectra_mask))

    def filter_custom(self, filter_array: np.ndarray) -> "Meteo":
        """
        Apply a custom boolean row filter to the meteorological data.

        Args:
            filter_array (np.ndarray): Boolean array with one entry per timestep.

        Returns:
            Meteo: A new Meteo instance with custom-filtered data.
        """
        return self._apply_row_mask(np.asarray(filter_array, dtype=bool))

    def reindex(self, index: pd.Index, method: str = "nearest", tolerance: pd.Timedelta | None = None) -> "Meteo":
        """
        Reindex the data according to the provided strictly increasing time indexer.

        All row-aligned attributes (pandas series/frames *and* the numpy
        jsc/bandgap/sigma/APE arrays) are aligned to 'index'; rows without a
        match within 'tolerance' become NaN. For regular target grids, trapezoid
        time weights 'dt' are recomputed on 'index'; when 'index' contains
        multi-step gaps (e.g. a daytime-only index), matched rows retain their
        original 'self.dt' weights so gaps are not bridged.
        """
        if tolerance is None:
            tolerance = pd.Timedelta(seconds=30)  # ty: ignore[invalid-assignment]

        index = pd.DatetimeIndex(index)
        if len(index) >= 2:
            idx_secs = (index - index[0]).total_seconds().to_numpy(dtype=np.float64)
            idx_gaps = np.diff(idx_secs)
            if np.any(idx_gaps <= 0.0):
                raise ValueError("Meteo.reindex: target index must be strictly increasing")
        else:
            idx_gaps = np.zeros(0, dtype=np.float64)

        self_copy = copy.deepcopy(self)

        # Reindex all pandas DataFrame or Series attributes
        for attr_name in vars(self):
            attr = getattr(self, attr_name)
            if (isinstance(attr, pd.DataFrame) or isinstance(attr, pd.Series)) and isinstance(attr.index, pd.DatetimeIndex):
                setattr(self_copy, attr_name, attr.reindex(index=index, method=method, tolerance=tolerance))
        # spectra keeps whatever row index the caller used; align it positionally
        if not isinstance(self.spectra.index, pd.DatetimeIndex):
            spectra_dt = self.spectra.set_index(self.datetime)
            self_copy.spectra = spectra_dt.reindex(index=index, method=method, tolerance=tolerance)

        # numpy row-aligned arrays: positional indexer, unmatched rows -> NaN
        indexer = self.datetime.get_indexer(index, method=method, tolerance=tolerance)
        for attr_name in ("jscs", "bandgaps", "sigmas", "average_photon_energy"):
            attr = getattr(self, attr_name)
            if attr is not None:
                new = np.asarray(attr, dtype=np.float64)[indexer]
                new[indexer < 0] = np.nan
                setattr(self_copy, attr_name, new)

        self_copy.datetime = index
        if len(index) >= 3 and float(np.median(idx_gaps)) > 0.0 and float(idx_gaps.max()) > 2.0 * float(np.median(idx_gaps)):
            logger.warning(
                "Meteo.reindex: target index has gaps up to {:.2f} h (median step {:.2f} h); preserving matched source dt weights instead of bridging gaps",
                float(idx_gaps.max()) / 3600.0,
                float(np.median(idx_gaps)) / 3600.0,
            )
            src_dt = self.dt.to_numpy(dtype=np.float64)
            w_out = np.where(indexer >= 0, src_dt[indexer], 0.0)
        else:
            w_out = _time_weights(index)
        self_copy.dt = pd.Series(w_out, index=index)
        self_copy._recompute_energy_in()
        return self_copy
