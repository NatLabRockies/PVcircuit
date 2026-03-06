import numpy as np
import pandas as pd
from scipy import constants
from typing import Union

# Constants
K_Q = constants.k / constants.e
HC_E = constants.h * constants.c / constants.e
DB_PREFIX = 2.0 * np.pi * constants.e * (constants.k / constants.h) ** 3 / (constants.c) ** 2 / 1.0e4  # about 1.0133e-8 for Jdb[A/cm2]


def TK(TC: float) -> float:
    """
    Convert temperature from Celsius to Kelvin.

    Args:
        TC (float): Temperature in Celsius.

    Returns:
        float: Temperature in Kelvin.
    """
    return TC + constants.zero_Celsius


def Vth(TC: float) -> float:
    """
    Calculate the thermal voltage.

    Args:
        TC (float): Temperature in Celsius.

    Returns:
        float: Thermal voltage.
    """
    return K_Q * TK(TC)


def wavelength_to_photonenergy(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert wavelength [nm] to photon energy [eV]

    Args:
        wavelength (Union[float, np.ndarray]): Wavelength in [nm]

    Returns:
        Union[float, np.ndarray]: Photon energy in [eV]
    """
    return HC_E / (wavelength * 1e-9)


def photonenergy_to_wavelength(photonenergy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert photon energy [eV] to wavelength [nm]

    Args:
        photonenergy (Union[float, np.ndarray]): Photon energy in [eV]

    Returns:
        Union[float, np.ndarray]: Wavelength in [nm]
    """
    return HC_E / (photonenergy * 1e-9)


def normalize(eqe: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the EQE data.

    Args:
        eqe (pd.DataFrame): EQE data.

    Returns:
        pd.DataFrame: Normalized EQE data.
    """
    eqe_min = eqe.min().min()
    eqe_max = eqe.max().max()
    return (eqe - eqe_min) / (eqe_max - eqe_min)
