# -*- coding: utf-8 -*-
"""
Package to simulate energy yield
"""

import copy
import multiprocessing as mp
import warnings
from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np  # arrays
import pandas as pd
from parse import parse
from scipy import constants
from scipy.integrate import trapezoid
from scipy.special import erfc
from tqdm import tqdm, trange

import pvcircuit as pvc

warnings.warn("The 'EY.py' module is deprecated and will be change in future version.", DeprecationWarning, stacklevel=2)


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

    elif isinstance(model, pvc.Tandem3T):  # Tandem3T
        tandem_type = oper.split("-")

        if tandem_type[0] == "MPP" or tandem_type[0] == "CM":
            return 1
        elif tandem_type[0] == "VM":
            if len(tandem_type) != 3:
                raise ValueError("3T voltage matched operation must be VM-[bc/tc ratio]-[r/s-type], e.g. VM-21-r")
            vm_ratio = tuple(map(int, tandem_type[1]))

            if tandem_type[2] == "r":
                endloss = max(vm_ratio) - 1
            elif tandem_type[2] == "s":
                endloss = sum(vm_ratio) - 1
            else:
                raise ValueError("Unknown model")
    else:
        raise ValueError("Unknown model")

    lossfactor = max(0, 1 - endloss / ncells)
    return lossfactor


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


def sandia_T(poa_global: float, wind_speed: float, temp_air: float) -> float:
    """
    Calculate the solar cell temperature using the Sandia model.

    Adapted from the pvlib library to avoid using pandas dataframes.
    Parameters used are those of 'open_rack_cell_polymerback'.

    Args:
        poa_global (float): Plane of array irradiance [W/m²].
        wind_speed (float): Wind speed [m/s].
        temp_air (float): Ambient air temperature [°C].

    Returns:
        float: Calculated cell temperature [°C].
    """
    a = -3.56
    b = -0.075
    deltaT = 3

    E0 = 1000.0  # Reference irradiance

    temp_module = poa_global * np.exp(a + b * wind_speed) + temp_air

    temp_cell = temp_module + (poa_global / E0) * (deltaT)

    return temp_cell


def _calc_yield_async(Jscs: np.ndarray, Egs: np.ndarray, sigmas: np.ndarray, TempCell: pd.Series, devlist: List[Union["pvc.Multi2T", "pvc.Tandem3T"]], oper: str) -> np.ndarray:
    Pmax_out = np.zeros(len(Jscs))

    for i in range(len(Jscs)):
        model = devlist[i]
        if isinstance(model, pvc.Multi2T):  # Multi2T or current matched 2-junction tandem
            for ijunc in range(model.njuncs):
                model.j[ijunc].set(Eg=Egs[ijunc], Jext=Jscs[i, ijunc] / 1e3, TC=TempCell[i])

            mpp_dict = model.MPP()
            Pmax = mpp_dict["Pmp"]

        elif isinstance(model, pvc.Tandem3T):  # Tandem3T
            tandem_type = oper.split("-")

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
                iv3T = pvc.iv3T.IV3T("bogus", shape=1)
                iv3T.Ptot[0] = 0
            Pmax = iv3T.Ptot[0]
        else:
            Pmax = 0.0

        Pmax_out[i] = Pmax

    return Pmax_out  # Pmax in [W]


class Meteo:
    """
    NOTE: All arrays in this class are handled so that each row aligns with a timestamp.
    For instance, any EQE array is assumed to correspond to the same timestamps as this data,
    and each row represents EQE values for that specific time index.
    """

    """
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
        self.cell_temp = sandia_T(self.irradiance, self.wind, self.temp)  # Cell temperature calculation
        self.energy_in = trapezoid(y=self.irradiance, x=self.datetime.astype(np.int64)) / 1e9 / 3600 / 1000  # Energy input [kWh/m²/yr]

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

        Args:
            jsc (np.ndarray): Short-circuit current values to add.
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
            Tuple[float, float]: A tuple containing energy yield [kWh/m²/yr] and energy harvesting efficiency.
        """
        # If sigma values are not provided, initialize them to zero
        if self.sigmas is None:
            self.sigmas = np.zeros_like(self.bandgaps)

        # Ensure all data arrays have consistent shapes
        for attr in [self.jscs, self.bandgaps, self.sigmas]:
            if attr is not None and attr.shape[0] != self.cell_temp.shape[0]:
                raise ValueError(f"Inconsistent array size: {attr.shape[0]} rows, expected {self.cell_temp.shape[0]}")

        # Determine chunk sizes for multiprocessing
        max_chunk_size = 200
        cpu_count = mp.cpu_count()
        chunk_ids = np.arange(len(self.jscs))
        chunk_size = min(len(chunk_ids) // cpu_count + 1, max_chunk_size)

        # Create chunks of data for parallel processing
        chunks = [chunk_ids[i : i + chunk_size] for i in range(0, len(chunk_ids), chunk_size)]
        dev_list = np.array([copy.deepcopy(model) for _ in range(len(self.jscs))])

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
                    jobs = [
                        pool.apply_async(_calc_yield_async, args=(self.jscs[chunk], self.bandgaps[chunk], self.sigmas[chunk], self.cell_temp.iloc[chunk], dev_list[chunk], oper), callback=update_tqdm)
                        for chunk in chunks
                    ]
                    # Collect results from workers
                    results = [item for job in jobs for item in job.get()]

            else:
                pbar.set_description(f"Running {model.name} in mode {oper} without multiprocessing")

                results = []
                for i, chunk in enumerate(chunks):
                    # Process each chunk sequentially
                    chunk_result = _calc_yield_async(self.jscs[chunk], self.bandgaps[chunk], self.sigmas[chunk], self.cell_temp.iloc[chunk], dev_list[chunk], oper)
                    results.extend(chunk_result)
                    pbar.update(len(chunk))
                    pbar.refresh()

        self.outPowerMP = results  # output power [W]

        power_density = self.outPowerMP / model.totalarea  # W --> W/cm²

        EnergyOut = trapezoid(power_density, self.datetime.values.astype(np.int64)) / 1e9  # [Ws/cm²/yr]

        EnergyOut = EnergyOut / 3.6e3 / 1e3 * 1e4  # [Ws/cm²/yr] --> [kWh/m²/yr]

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

        self_copy = copy.deepcopy(self)
        # Create a mask based on APE criteria
        ape_mask = (self_copy.average_photon_energy > min_ape) & (self_copy.average_photon_energy < max_ape)

        # Apply the mask to relevant attributes
        self_copy.datetime = self_copy.datetime[ape_mask]
        self_copy.average_photon_energy = self_copy.average_photon_energy[ape_mask]
        self_copy.spectra = self_copy.spectra[ape_mask]
        self_copy.irradiance = self_copy.irradiance[ape_mask]
        self_copy.cell_temp = self_copy.cell_temp[ape_mask]

        # Ensure all filtered attributes have the same length
        assert len(self_copy.spectra) == len(self_copy.irradiance) == len(self_copy.cell_temp) == len(self_copy.average_photon_energy)
        return self_copy

    def filter_spectra(self, min_spectra: float = 0, max_spectra: float = 10) -> "Meteo":
        """
        Filter the spectral data within specified bounds.

        Args:
            min_spectra (float, optional): Minimum value of the spectra. Defaults to 0.
            max_spectra (float, optional): Maximum value of the spectra. Defaults to 10.

        Returns:
            Meteo: A new Meteo instance with filtered spectral data.
        """
        self_copy = copy.deepcopy(self)
        # Create a mask based on spectral criteria
        spectra_mask = (self_copy.spectra >= min_spectra).all(axis=1) & (self_copy.spectra < max_spectra).all(axis=1)
        # Apply the mask to relevant attributes
        self_copy.datetime = self_copy.datetime[spectra_mask]
        self_copy.average_photon_energy = self_copy.average_photon_energy[spectra_mask]
        self_copy.spectra = self_copy.spectra[spectra_mask]
        self_copy.irradiance = self_copy.irradiance[spectra_mask]
        self_copy.cell_temp = self_copy.cell_temp[spectra_mask]

        # Ensure all filtered attributes have the same length
        assert len(self.spectra) == len(self.irradiance) == len(self.cell_temp) == len(self.average_photon_energy)
        return self_copy

    def filter_custom(self, filter_array: np.ndarray) -> "Meteo":
        """
        Apply a custom filter to the meteorological data.

        Args:
            filter_array (np.ndarray): Boolean array used to filter the data.

        Returns:
            Meteo: A new Meteo instance with custom-filtered data.
        """
        # assert len(filter_array) == len(self.spectra) == len(self.SpecPower) == len(self.TempCell)

        self_copy = copy.deepcopy(self)

        # Apply the filter to all relevant attributes
        for attr_name in vars(self):
            if hasattr(getattr(self_copy, attr_name), "__len__"):
                attr = getattr(self_copy, attr_name)
                if len(attr) == len(filter_array):
                    setattr(self_copy, attr_name, attr[filter_array])

        # assert len(self.spectra) == len(self.SpecPower) == len(self.TempCell) == len(self.average_photon_energy)
        return self_copy

    def reindex(self, index: pd.Index, method: str = "nearest", tolerance: pd.Timedelta = pd.Timedelta(seconds=30)) -> "Meteo":
        """
        Reindex the data according to the provided time indexer.

        Args:
            index (pd.Index): New index to reindex the data to.
            method (str, optional): Method to use for reindexing. Defaults to "nearest".
            tolerance (pd.Timedelta, optional): Tolerance for the nearest method. Defaults to 30 seconds.

        Returns:
            Meteo: A new Meteo instance with reindexed data.
        """
        self_copy = copy.deepcopy(self)

        # Reindex all pandas DataFrame or Series attributes
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if (isinstance(attr, pd.DataFrame) or isinstance(attr, pd.Series)) and isinstance(attr.index, pd.DatetimeIndex):
                setattr(self_copy, attr_name, attr.reindex(index=index, method=method, tolerance=tolerance))
        self_copy.datetime = index
        return self_copy
