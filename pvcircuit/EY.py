# -*- coding: utf-8 -*-
"""
Package to simulate energy yield
"""

import copy
import multiprocessing as mp
from typing import List, Optional, Tuple, Union

import numpy as np  # arrays
import pandas as pd
from scipy import constants
from scipy.integrate import trapezoid
from tqdm import tqdm

import pvcircuit as pvc


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


def _calc_yield_async(Jscs: np.ndarray, Egs: np.ndarray, sigmas: np.ndarray, TempCell: pd.Series, model: Union["pvc.Multi2T", "pvc.Tandem3T"], oper: str) -> pd.DataFrame:
    """Evaluate IV parameters for one chunk of timesteps on a single device copy.

    The same model instance is reused for every row: each iteration overwrites
    all state that varies between rows (Eg, sigma, Jext, TC), so no state leaks
    from one timestep to the next.
    """

    columns: list[str] = ["Voc", "Isc", "Vmp", "Imp", "Pmp"]
    IV_params = pd.DataFrame(np.zeros((len(Jscs), len(columns))), columns=columns)

    for i in range(len(Jscs)):
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
            IV_params.loc[i, "Vmp"] = iv3T.VA - iv3T.VB
            IV_params.loc[i, "Imp"] = min(abs(iv3T.IA), abs(iv3T.IB))
            IV_params.loc[i, "Pmp"] = iv3T.Ptot

            iv3T = model.Voc3()
            IV_params.loc[i, "Voc"] = iv3T.VA - iv3T.VB

            iv3T = model.Isc3()
            IV_params.loc[i, "Isc"] = min(abs(iv3T.IA), abs(iv3T.IB))

        else:
            raise ValueError(f"Unknown model type: {type(model).__name__}")

    return IV_params  # Pmax in [W]


class Meteo:
    """
    NOTE: All arrays in this class are handled so that each row aligns with a timestamp.
    For instance, any EQE array is assumed to correspond to the same timestamps as this data,
    and each row represents EQE values for that specific time index.
    Handles meteorological environmental data and spectral information for energy yield simulations.
    """

    def __init__(self, wavelength: np.ndarray, spectra: pd.DataFrame, ambient_temperature: pd.Series, wind: pd.Series, datetime: pd.DatetimeIndex) -> None:
        # Replace NaN values in spectra with 0 to ensure data integrity
        spectra = spectra.fillna(0)
        # Create a filter to drop any remaining NaNs in ambient_temperature or wind
        ffilter = (np.all(np.isfinite(spectra), axis=1)) & (np.isfinite(ambient_temperature)) & (np.isfinite(wind))
        self.temp = ambient_temperature[ffilter]  # Ambient temperature in degrees Celsius
        self.wind = wind[ffilter]  # Wind speed in meters per second
        self.datetime = datetime[ffilter]  # Filtered datetime index

        self.wavelength = wavelength
        self.spectra = spectra[ffilter]  # Spectral data after filtering

        # Calculate irradiance from spectral proxy data
        self.irradiance = pd.Series(trapezoid(y=self.spectra, x=self.wavelength), index=self.datetime)  # Optical power of each spectrum
        self.cell_temp: pd.Series = pd.Series(
            sandia_T(self.irradiance.to_numpy(), self.wind.to_numpy(), self.temp.to_numpy()),
            index=self.datetime,
        )  # Cell temperature calculation
        # Convert datetimes to float seconds since the first sample.
        # `(datetime - datetime[0]).total_seconds()` is pandas-native and
        # resolution-agnostic, so it works under both pandas 2.x ([ns])
        # and pandas 3.x ([us]).  Using `astype(np.int64)/1e9` would be
        # silently 1000x off under pandas >= 3.0.
        self.energy_in = trapezoid(y=self.irradiance, x=(self.datetime - self.datetime[0]).total_seconds()) / 3600 / 1000  # Energy input [kWh/m^2/yr]

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
        # Ensure the array is columnar and matches the number of rows in cell_temp
        if array.ndim == 1:
            array = array[:, np.newaxis]
        elif array.shape[1] == self.cell_temp.shape[0] and array.shape[0] == 1:
            array = array.T

        if array.shape[0] != self.cell_temp.shape[0]:
            raise ValueError(f"Shape of data {array.shape} does not match cell_temp rows {self.cell_temp.shape[0]}")

        current_attr = getattr(self, attribute_name)
        if current_attr is None:
            setattr(self, attribute_name, array)
        else:
            setattr(self, attribute_name, np.concatenate((current_attr, array), axis=1))

    def add_currents(self, jsc: np.ndarray) -> None:
        """
        Add Jsc array to the instance.

        Jsc values are stored in self.jscs in **mA/cm^2** (the native unit
        of JintMD).  They are converted to A/cm^2 inside
        _calc_yield_async before being assigned to
        Junction.Jext.

        Args:
            jsc (np.ndarray): Short-circuit current values to add [mA/cm^2].
        """
        self._add_array(jsc, "jscs")

    def add_bandgaps(self, bandgap: np.ndarray) -> None:
        """
        Add bandgap array to the instance.

        Args:
            bandgap (np.ndarray): Bandgap energy values to add.
        """
        self._add_array(bandgap, "bandgaps")

    def add_sigmas(self, sigma: np.ndarray) -> None:
        """
        Add sigma array for bandgap tail states to the instance.

        Args:
            sigma (np.ndarray): Sigma values to add.
        """
        self._add_array(sigma, "sigmas")

    def run_ey(self, model: Union["pvc.Multi2T", "pvc.Tandem3T"], oper: str, multiprocessing: bool = True) -> Tuple[float, float]:
        """
        Calculate the energy yield and efficiency based on the provided model and operation mode.

        Args:
            model (Union["pvc.Multi2T", "pvc.Tandem3T"]): Either a Multi2T or Tandem3T model.
            oper (str): Operation mode, e.g., 'MPP', 'CM', 'VM-21-r', 'VM-21-s'.
            multiprocessing (bool, optional): Whether to use multiprocessing. Defaults to True.

        Raises:
            ValueError: If data array sizes are inconsistent with cell temperature.

        Returns:
            Tuple[float, float]: A tuple containing energy yield [kWh/m^2/yr] and energy harvesting efficiency.
        """
        # If sigma values are not provided, initialize them to zero
        if self.sigmas is None:
            self.sigmas = np.zeros_like(self.bandgaps)

        assert self.jscs is not None
        assert self.bandgaps is not None
        assert self.sigmas is not None

        # Ensure all data arrays have consistent shapes
        for attr in [self.jscs, self.bandgaps, self.sigmas]:
            if attr.shape[0] != self.cell_temp.shape[0]:
                raise ValueError(f"Inconsistent array size: {attr.shape[0]} rows, expected {self.cell_temp.shape[0]}")

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

        power = results["Pmp"]  # output power [W]

        # Normalise by device.totalarea (= max junction totalarea). This is
        # the project-wide reference area for power normalisation: it matches
        # how Multi2T.Rs2T is consumed in V2T and what Multi2T.efficiency()
        # and Tandem3T.efficiency() use, so kWh/m^2/yr corresponds to the
        # device footprint a system installer would see.
        power_density = power / model.totalarea  # W --> W/cm^2

        EnergyOut = trapezoid(power_density, (self.datetime - self.datetime[0]).total_seconds())  # [Ws/cm^2/yr]

        # Unit conversion breakdown for the factor /3.6e3 /1e3 *1e4:
        #   /3600  : seconds -> hours      (Ws -> Wh)
        #   /1000  : W -> kW               (Wh -> kWh)
        #   *1e4   : cm^-2 -> m^-2            (per-cm^2 -> per-m^2, since 1 m^2 = 1e4 cm^2)
        EnergyOut = EnergyOut / 3.6e3 / 1e3 * 1e4  # [Ws/cm^2/yr] --> [kWh/m^2/yr]

        # Calculate energy harvesting efficiency
        EYeff = EnergyOut / self.energy_in
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
        """Recompute the integrated energy input from the current (filtered) rows."""
        if len(self.datetime) == 0:
            self.energy_in = 0.0
            return
        self.energy_in = trapezoid(y=self.irradiance, x=(self.datetime - self.datetime[0]).total_seconds()) / 3600 / 1000  # [kWh/m^2/yr]

    def _apply_row_mask(self, mask: np.ndarray) -> "Meteo":
        """
        Return a deep copy with a boolean row mask applied to every
        row-aligned attribute (meteorological series and any added
        jsc/bandgap/sigma arrays), and energy_in recomputed.
        """
        self_copy = copy.deepcopy(self)

        self_copy.datetime = self_copy.datetime[mask]
        self_copy.temp = self_copy.temp[mask]
        self_copy.wind = self_copy.wind[mask]
        self_copy.spectra = self_copy.spectra[mask]
        self_copy.irradiance = self_copy.irradiance[mask]
        self_copy.cell_temp = self_copy.cell_temp[mask]
        if self_copy.average_photon_energy is not None:
            self_copy.average_photon_energy = self_copy.average_photon_energy[mask]
        for attr_name in ("jscs", "bandgaps", "sigmas"):
            attr = getattr(self_copy, attr_name)
            if attr is not None:
                setattr(self_copy, attr_name, attr[mask])

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
        Apply a custom filter to the meteorological data.

        Args:
            filter_array (np.ndarray): Boolean array used to filter the data.

        Returns:
            Meteo: A new Meteo instance with custom-filtered data.
        """
        self_copy = copy.deepcopy(self)

        # Apply the filter to all row-aligned attributes
        for attr_name in vars(self):
            if hasattr(getattr(self_copy, attr_name), "__len__"):
                attr = getattr(self_copy, attr_name)
                if len(attr) == len(filter_array):
                    setattr(self_copy, attr_name, attr[filter_array])

        self_copy._recompute_energy_in()
        return self_copy

    def reindex(self, index: pd.Index, method: str = "nearest", tolerance: Optional[pd.Timedelta] = None) -> "Meteo":
        """
        Reindex the data according to the provided time indexer.

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

        self_copy = copy.deepcopy(self)

        # Reindex all pandas DataFrame or Series attributes
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if (isinstance(attr, pd.DataFrame) or isinstance(attr, pd.Series)) and isinstance(attr.index, pd.DatetimeIndex):
                setattr(self_copy, attr_name, attr.reindex(index=index, method=method, tolerance=tolerance))
        self_copy.datetime = index
        return self_copy
