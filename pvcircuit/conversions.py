import numpy as np
import pandas as pd
from scipy import constants
from typing import Union

# Physical constants (see Conventions section in copilot-instructions.md for the
# full unit catalogue used across PVcircuit).
K_Q = constants.k / constants.e
"""[V/K] Boltzmann constant divided by elementary charge. Used in :func:'Vth'
to compute the thermal voltage kT/q."""

HC_E = constants.h * constants.c / constants.e
"""[V*m] Planck constant times speed of light divided by elementary charge.
Divide by (wavelength * 1e-9) to get the photon energy in eV when the
wavelength is supplied in nm; equivalently 'HC_E * 1e9' \approx 1239.84 eV*nm."""

DB_PREFIX = 2.0 * np.pi * constants.e * (constants.k / constants.h) ** 3 / (constants.c) ** 2 / 1.0e4
r"""[A/(cm^2*K^3)] Detailed-balance prefactor 2\pi*e*(k/h)^3/c^2 with the trailing
'/1e4' converting m^2 -> cm^2. Used in :func:'pvcircuit.junction.Jdb' to compute
the radiative saturation current density. Value \approx 1.0133e-8 A/(cm^2*K^3)."""


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
    Normalize the EQE data to the range [0, 1].

    Args:
        eqe (pd.DataFrame): EQE data.

    Returns:
        pd.DataFrame: Normalized EQE data. If the input is constant
        (max == min) the original input is returned unchanged to avoid
        division by zero.
    """
    eqe_min = eqe.min().min()
    eqe_max = eqe.max().max()
    span = eqe_max - eqe_min
    if span == 0:
        return eqe - eqe_min  # all zeros, preserves shape and type
    return (eqe - eqe_min) / span
