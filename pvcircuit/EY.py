# -*- coding: utf-8 -*-
"""
Package to simulate energy yield
"""

import copy
import multiprocessing as mp
from typing import List, Optional, Tuple, Union

import numpy as np  # arrays
import pandas as pd
from loguru import logger
from scipy import constants
from scipy.integrate import trapezoid
from tqdm import tqdm

import pvcircuit as pvc

# Below this upper wavelength bound the integrated spectra can no longer be
# regarded as broadband plane-of-array irradiance (AM1.5G carries ~5 % of its
# power above 2500 nm) -- cell temperature and energy_in would be biased.
_MIN_BROADBAND_WAVELENGTH_NM = 2500.0


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
def VMlist(mmax: int) -> List[str]:
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


def sandia_T(poa_global: Union[float, np.ndarray, pd.Series], wind_speed: Union[float, np.ndarray, pd.Series], temp_air: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
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
    Jsc produce zero output (Pmp = Isc = Imp = 0, Voc = Vmp = 0). Multi2T.MPP()
    returns NaN at zero photocurrent, which would otherwise poison the time
    integral of the whole year.
    """

    columns: list[str] = ["Voc", "Isc", "Vmp", "Imp", "Pmp"]
    IV_params = pd.DataFrame(np.zeros((len(Jscs), len(columns))), columns=columns)

    for i in range(len(Jscs)):
        row_jsc = np.asarray(Jscs[i], dtype=np.float64)
        # Same threshold as Multi2T.MPP (1e-6 A/cm^2 = 1e-3 mA/cm^2), which
        # returns NaN below it.
        if not np.all(np.isfinite(row_jsc)) or np.max(row_jsc) <= 1e-3:
            # zero output; the DataFrame is pre-filled with zeros
            continue

        if isinstance(model, pvc.Multi2T):  # Multi2T or current matched 2-junction tandem
            for ijunc in range(model.njuncs):
                # Jscs is stored in mA/cm^2 (Meteo.add_currents); junction.Jext
                # expects A/cm^2, so divide by 1000.
                model.j[ijunc].set(Eg=Egs[i, ijunc], sigma=sigmas[i, ijunc], Jext=Jscs[i, ijunc] / 1e3, TC=TempCell.iloc[i])

            mpp_dict = model.MPP()
            # Pmax = mpp_dict["Pmp"]
            for col in columns:
                IV_params.loc[i, col] = mpp_dict[col]

        elif isinstance(model, pvc.Tandem3T):  # Tandem3T
            tandem_type = oper.split("-")

            # Jscs in mA/cm^2 -> A/cm^2 for junction.Jext, same conversion as above.
            model.top.set(Eg=Egs[i, 0], sigma=sigmas[i, 0], Jext=Jscs[i, 0] / 1e3, TC=TempCell.iloc[i])
            model.bot.set(Eg=Egs[i, 1], sigma=sigmas[i, 1], Jext=Jscs[i, 1] / 1e3, TC=TempCell.iloc[i])
            if tandem_type[0] == "MPP":
                tempRz = model.Rz
                model.set(Rz=0)
                iv3T = model.MPP()
                model.set(Rz=tempRz)
            elif tandem_type[0] == "CM":
                ln, iv3T = model.CM()
            elif tandem_type[0] == "VM":
                if len(tandem_type) != 3:
                    raise ValueError("3T voltage matched operation must be VM-[bc/tc ratio]-[r/s-type], e.g. VM-21-r")
                model.bot.pn = -1 * model.top.pn if tandem_type[2] == "r" else 1 * model.top.pn
                ln, iv3T = model.VM(*map(int, tandem_type[1]))
            else:
                raise ValueError(f"Unknown 3T operation mode: {oper!r}")
            assert isinstance(iv3T, pvc.iv3T.IV3T)
            # Load-terminal quantities: VA - VB and min(|IA|, |IB|) equal the
            # tandem Vmp/Imp (and Voc/Isc below) for CM operation; for MPP/VM they
            # are the terminal values of the load configuration. Pmp (Ptot) is
            # the total device output in every mode.
            IV_params.loc[i, "Vmp"] = iv3T.VA - iv3T.VB
            IV_params.loc[i, "Imp"] = min(abs(iv3T.IA), abs(iv3T.IB))
            IV_params.loc[i, "Pmp"] = iv3T.Ptot

            iv3T = model.Voc3()
            IV_params.loc[i, "Voc"] = iv3T.VA - iv3T.VB

            iv3T = model.Isc3()
            IV_params.loc[i, "Isc"] = min(abs(iv3T.IA), abs(iv3T.IB))

        else:
            raise ValueError(f"Unknown model type: {type(model).__name__}")

    # Any remaining non-finite value (e.g. a solver failure on a single row)
    # must not poison the yield integral.
    if not np.all(np.isfinite(IV_params.to_numpy())):
        nbad = int((~np.isfinite(IV_params.to_numpy())).any(axis=1).sum())
        logger.warning("_calc_yield_async: {} of {} timesteps returned non-finite IV parameters; set to 0", nbad, len(IV_params))
        IV_params = IV_params.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    return IV_params  # Pmax in [W]


class Meteo:
    """
    Handles meteorological environmental data and spectral information for energy yield simulations.

    NOTE: All arrays in this class are handled so that each row aligns with a timestamp.
    For instance, any EQE array is assumed to correspond to the same timestamps as this data,
    and each row represents EQE values for that specific time index.

    Row-aligned attributes: ``datetime``, ``temp``, ``wind``, ``spectra``, ``irradiance``,
    ``cell_temp``, ``dt`` (integration weight per row [s]), ``average_photon_energy``,
    ``jscs``, ``bandgaps``, ``sigmas`` and, after :meth:`run_ey`, ``results``.

    Time integration (``energy_in``, energy yield) uses the per-row weights ``dt``
    (trapezoid weights of the original timeline). They are kept when rows are
    filtered, so filtered copies integrate only the rows they contain.

    Spectra: NaN entries are replaced by 0 (a partially missing spectrum still
    contributes its remaining bands to the irradiance). Feed the *filled* spectra
    (``Meteo.spectra``, ``Meteo.wavelength``) to :meth:`EQE.add_spectra` so that
    the Jsc time series stays aligned with ``cell_temp`` and does not contain NaN.
    """

    def __init__(self, wavelength: np.ndarray, spectra: pd.DataFrame, ambient_temperature: pd.Series, wind: pd.Series, datetime: pd.DatetimeIndex) -> None:
        wavelength = np.asarray(wavelength, dtype=np.float64).flatten()
        spectra = pd.DataFrame(spectra)
        ambient_temperature = pd.Series(np.asarray(ambient_temperature, dtype=np.float64))
        wind = pd.Series(np.asarray(wind, dtype=np.float64))
        datetime = pd.DatetimeIndex(datetime)

        nrows = len(datetime)
        for name, length in (("spectra", spectra.shape[0]), ("ambient_temperature", len(ambient_temperature)), ("wind", len(wind))):
            if length != nrows:
                raise ValueError(f"Meteo: {name} has {length} rows but datetime has {nrows} entries; all inputs must be row-aligned")
        if spectra.shape[1] != len(wavelength):
            raise ValueError(f"Meteo: spectra has {spectra.shape[1]} columns but wavelength has {len(wavelength)} entries; spectra must be (n_times, n_wavelengths)")

        # Replace NaN values in spectra with 0 to ensure data integrity
        n_nan_rows = int(spectra.isna().any(axis=1).sum())
        if n_nan_rows > 0:
            logger.warning(
                "Meteo: {} of {} spectra rows contain NaN; they are filled with 0 (partial irradiance still counts). Use Meteo.spectra for EQE.add_spectra so Jsc has no NaN.",
                n_nan_rows,
                nrows,
            )
        spectra = spectra.fillna(0)
        # Create a filter to drop any remaining NaNs in ambient_temperature or wind
        # (positional numpy masks: the pandas indexes of the inputs need not agree)
        ffilter = np.all(np.isfinite(spectra.to_numpy()), axis=1) & np.isfinite(ambient_temperature.to_numpy()) & np.isfinite(wind.to_numpy())
        if not ffilter.all():
            logger.warning("Meteo: dropping {} rows with non-finite temperature/wind", int((~ffilter).sum()))
        # `spectra` keeps its own row index (whatever the caller used); the
        # datetime is kept separately in self.datetime.
        self.temp = pd.Series(ambient_temperature.to_numpy()[ffilter], index=datetime[ffilter])  # Ambient temperature in degrees Celsius
        self.wind = pd.Series(wind.to_numpy()[ffilter], index=datetime[ffilter])  # Wind speed in meters per second
        self.datetime = datetime[ffilter]  # Filtered datetime index

        self.wavelength = wavelength
        self.spectra = spectra.iloc[ffilter]  # Spectral data after filtering

        if wavelength.max() < _MIN_BROADBAND_WAVELENGTH_NM:
            logger.warning(
                "Meteo: spectra only extend to {:.0f} nm. Irradiance (cell temperature) and energy_in are integrated over the given range only; pass broadband spectra (~280-4000 nm) for a correct energy harvesting efficiency.",
                wavelength.max(),
            )

        # Calculate irradiance from spectral proxy data
        self.irradiance = pd.Series(trapezoid(y=self.spectra.to_numpy(), x=self.wavelength), index=self.datetime)  # Optical power of each spectrum [W/m^2]
        self.cell_temp: pd.Series = pd.Series(
            sandia_T(self.irradiance.to_numpy(), self.wind.to_numpy(), self.temp.to_numpy()),
            index=self.datetime,
        )  # Cell temperature calculation
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

    def _add_array(self, array: np.ndarray, attribute_name: str) -> None:
        """
        Helper function to add data arrays (e.g., jsc, bandgap, sigma) to the instance.

        Args:
            array (np.ndarray): Array to add.
            attribute_name (str): Name of the attribute to which the array will be added.
        """
        array = np.asarray(array, dtype=np.float64)
        # Ensure the array is columnar and matches the number of rows in cell_temp
        if array.ndim == 1:
            array = array[:, np.newaxis]
        elif array.shape[1] == self.cell_temp.shape[0] and array.shape[0] == 1:
            array = array.T

        if array.shape[0] != self.cell_temp.shape[0]:
            raise ValueError(f"Shape of data {array.shape} does not match cell_temp rows {self.cell_temp.shape[0]}")

        if not np.all(np.isfinite(array)):
            logger.warning(
                "Meteo.{}: {} of {} rows contain non-finite values; those timesteps will yield zero output in run_ey",
                "add_" + {"jscs": "currents", "bandgaps": "bandgaps", "sigmas": "sigmas"}.get(attribute_name, attribute_name),
                int((~np.isfinite(array)).any(axis=1).sum()),
                array.shape[0],
            )

        current_attr = getattr(self, attribute_name)
        if current_attr is None:
            setattr(self, attribute_name, array)
        else:
            setattr(self, attribute_name, np.concatenate((current_attr, array), axis=1))

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

    def run_ey(self, model: Union["pvc.Multi2T", "pvc.Tandem3T"], oper: str, multiprocessing: bool = True) -> Tuple[float, float]:
        """
        Calculate the energy yield and efficiency based on the provided model and operation mode.

        Args:
            model (Union["pvc.Multi2T", "pvc.Tandem3T"]): Either a Multi2T or Tandem3T model.
            oper (str): Operation mode, e.g., 'MPP', 'CM', 'VM-21-r', 'VM-21-s'.
            multiprocessing (bool, optional): Whether to use multiprocessing. Defaults to True.

        Raises:
            ValueError: If data array sizes are inconsistent with cell temperature, or if
                the number of Jsc/bandgap/sigma columns does not match the number of
                junctions of ``model``.

        Returns:
            Tuple[float, float]: A tuple containing energy yield [kWh/m^2/yr] and energy harvesting efficiency.
        """
        if self.jscs is None or self.bandgaps is None:
            raise ValueError("run_ey: add_currents() and add_bandgaps() must be called before run_ey()")

        # If sigma values are not provided, initialize them to zero
        if self.sigmas is None:
            self.sigmas = np.zeros_like(self.bandgaps)

        assert self.jscs is not None
        assert self.bandgaps is not None
        assert self.sigmas is not None

        # Ensure all data arrays have consistent shapes
        njuncs = self._model_njuncs(model)
        for name, attr in (("jscs", self.jscs), ("bandgaps", self.bandgaps), ("sigmas", self.sigmas)):
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

        with tqdm(total=len(self.datetime), leave=True) as pbar:

            def update_tqdm(*args):
                """Callback function to update the progress bar."""
                pbar.update(len(args[0]))
                pbar.refresh()
                return

            if multiprocessing:
                pbar.set_description(f"Running {model.name} in mode {oper} with {cpu_count} processes")

                with mp.Pool(cpu_count) as pool:
                    # Assign tasks to multiprocessing pool
                    # For multiprocessing
                    jobs = [
                        pool.apply_async(_calc_yield_async, args=(self.jscs[chunk], self.bandgaps[chunk], self.sigmas[chunk], self.cell_temp.iloc[chunk], copy.deepcopy(model), oper), callback=update_tqdm)
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
                    chunk_result = _calc_yield_async(self.jscs[chunk], self.bandgaps[chunk], self.sigmas[chunk], self.cell_temp.iloc[chunk], copy.deepcopy(model), oper)
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
        EnergyOut = float(np.sum(power_density * self.dt.to_numpy()))  # [Ws/cm^2/yr]

        # Unit conversion breakdown for the factor /3.6e3 /1e3 *1e4:
        #   /3600  : seconds -> hours      (Ws -> Wh)
        #   /1000  : W -> kW               (Wh -> kWh)
        #   *1e4   : cm^-2 -> m^-2            (per-cm^2 -> per-m^2, since 1 m^2 = 1e4 cm^2)
        EnergyOut = EnergyOut / 3.6e3 / 1e3 * 1e4  # [Ws/cm^2/yr] --> [kWh/m^2/yr]

        # Calculate energy harvesting efficiency
        EYeff = EnergyOut / self.energy_in if self.energy_in > 0 else np.nan
        return EnergyOut, EYeff

    def calc_ape(self) -> None:
        """
        Calculate the average photon energy (APE) of the spectra.
        """
        # Calculate photon flux
        phi = self.spectra * (self.wavelength * 1e-9) / constants.h / constants.c
        # Identify and mask rows where all photon flux values are zero
        mask = (phi == 0).all(axis=1)
        phi[mask] = np.nan
        # Compute average photon energy
        self.average_photon_energy = trapezoid(x=self.wavelength, y=self.spectra.values) / constants.e / trapezoid(x=self.wavelength, y=phi.values)

    def _recompute_energy_in(self) -> None:
        """Recompute the integrated energy input [kWh/m^2/yr] from the current rows and their time weights."""
        if len(self.datetime) == 0:
            self.energy_in = 0.0
            return
        self.energy_in = float(np.nansum(self.irradiance.to_numpy() * self.dt.to_numpy())) / 3600 / 1000  # [Ws/m^2] -> [kWh/m^2/yr]

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
        if getattr(self_copy, "results", None) is not None:
            self_copy.results = self_copy.results[mask]

        # energy_in was integrated over the unfiltered rows; recompute so
        # run_ey efficiency on the filtered copy is normalised correctly.
        self_copy._recompute_energy_in()
        return self_copy

    def filter_ape(self, min_ape: float = 0, max_ape: float = 10) -> "Meteo":
        """
        Filter the average photon energy (APE) within specified bounds.

        Args:
            min_ape (float, optional): Minimum value of the APE. Defaults to 0.
            max_ape (float, optional): Maximum value of the APE. Defaults to 10.

        Returns:
            Meteo: A new Meteo instance with filtered data.
        """
        if self.average_photon_energy is None:
            self.calc_ape()
        assert self.average_photon_energy is not None

        # NaN APE rows (all-zero spectra) compare False and are dropped
        ape_mask = (self.average_photon_energy > min_ape) & (self.average_photon_energy < max_ape)
        return self._apply_row_mask(np.asarray(ape_mask))

    def filter_spectra(self, min_spectra: float = 0, max_spectra: float = 10) -> "Meteo":
        """
        Filter the spectral data within specified bounds.

        Args:
            min_spectra (float, optional): Minimum value of the spectra. Defaults to 0.
            max_spectra (float, optional): Maximum value of the spectra. Defaults to 10.

        Returns:
            Meteo: A new Meteo instance with filtered spectral data.
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

    def reindex(self, index: pd.Index, method: str = "nearest", tolerance: Optional[pd.Timedelta] = None) -> "Meteo":
        """
        Reindex the data according to the provided time indexer.

        All row-aligned attributes (pandas series/frames *and* the numpy
        jsc/bandgap/sigma/APE arrays) are aligned to ``index``; rows without a
        match within ``tolerance`` become NaN. Time weights ``dt`` and
        ``energy_in`` are recomputed for the new index.

        Args:
            index (pd.Index): New index to reindex the data to.
            method (str, optional): Method to use for reindexing. Defaults to "nearest".
            tolerance (pd.Timedelta, optional): Tolerance for the nearest method. Defaults to 30 seconds.

        Returns:
            Meteo: A new Meteo instance with reindexed data.
        """
        if tolerance is None:
            # pandas stubs widen Timedelta(...) to Timedelta | NaTType even though
            # this literal call cannot produce NaT.
            tolerance = pd.Timedelta(seconds=30)  # ty: ignore[invalid-assignment]

        index = pd.DatetimeIndex(index)
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
        self_copy.dt = pd.Series(_time_weights(index), index=index)
        self_copy._recompute_energy_in()
        return self_copy
