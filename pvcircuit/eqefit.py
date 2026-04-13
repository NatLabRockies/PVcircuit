# -*- coding: utf-8 -*-
"""
pvcircuit.eqefit
~~~~~~~~~~~~~~~~
Temperature-dependent EQE reconstruction from per-subcell Voc(T) and Jsc(T).

Inputs:
    - Reference EQE at 25 °C per subcell
    - Per-subcell Voc(T), Jsc(T) as arrays over temperature
    - Per-subcell diode parameters: ideality factor n, J0 at 25 °C [mA/cm²]

Outputs:
    - EQE(T) arrays of shape (n_wavelength, n_temperatures)

Units throughout:
    - Wavelength: nm
    - Energy: eV
    - Current density: mA/cm²
    - Voltage: V
    - Temperature: °C (TC) unless noted

Independent of the Holovsky/lcfit subcell extraction workflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import erfc

from pvcircuit.conversions import TK, Vth
from pvcircuit.qe import JdbFromEg, nm2eV


@dataclass
class ErfcTailParams:
    """Parameters for the erfc EQE tail model."""
    Eg: float     # bandgap [eV]  — fixed from reference EQE
    A: float      # tail amplitude (≈ EQE plateau value)
    sigma: float  # Gaussian broadening [eV]


@dataclass
class OscillatorParams:
    """Parameters for Tauc-Lorentz or Elliott EQE tail model."""
    model: str         # 'tauc_lorentz' or 'elliott'
    Eg: float          # bandgap [eV]
    A: float           # primary amplitude (continuum for Elliott)
    E0: float          # resonance energy (TL) or exciton binding energy Eb (Elliott) [eV]
    Gamma: float       # broadening at T_ref [eV]
    A2: float = 0.0    # exciton amplitude (Elliott only)
    Gamma_ph: float = 0.0  # phonon coupling [eV]; 0 = 'fixed' broadening model


def _tauc_lorentz(E: np.ndarray, Eg: float, A: float, E0: float, Gamma: float) -> np.ndarray:
    """Tauc-Lorentz EQE model. Returns 0 for E <= Eg."""
    tl = np.where(
        E > Eg,
        A * Gamma * (E - Eg)**2 / ((E**2 - E0**2)**2 + Gamma**2 * E**2),
        0.0,
    )
    return np.clip(tl, 0.0, None)


def _elliott(E: np.ndarray, Eg: float, A: float, E0: float, Gamma: float, A2: float) -> np.ndarray:
    """Simplified Elliott model: Lorentzian exciton peak + erfc continuum.

    Parameters
    ----------
    E0 : exciton binding energy Eb [eV] — exciton peak sits at Eg - Eb
    A2 : exciton peak amplitude
    A  : continuum amplitude
    """
    exciton = A2 * Gamma / ((E - (Eg - E0))**2 + Gamma**2)
    continuum = A * 0.5 * erfc((Eg - E) / (Gamma * np.sqrt(2)))
    return np.clip(exciton + continuum, 0.0, None)


def extract_eg_from_voc(
    TC_arr: np.ndarray,
    Voc_arr: np.ndarray,
    Jsc_arr: np.ndarray,
    n: float,
    J0_ref: float,      # mA/cm²  (same units as Jsc_arr)
    Eg_ref: float,      # eV — bandgap at T_ref from EQE fit
    T_ref: float = 25.0,
) -> np.ndarray:
    """
    Invert per-subcell Voc(T) and Jsc(T) to extract Eg(T).

    Uses the single-diode relation:
        n·kT·ln(Jsc · η_ext / (JdbFromEg(T, Eg)·1000)) = Voc
    where η_ext = JdbFromEg(T_ref, Eg_ref)·1000 / J0_ref absorbs
    non-radiative losses (assumed T-independent).

    Parameters
    ----------
    TC_arr : array of temperatures [°C]
    Voc_arr : array of open-circuit voltages [V], same length as TC_arr
    Jsc_arr : array of short-circuit current densities [mA/cm²]
    n : ideality factor
    J0_ref : dark saturation current at T_ref [mA/cm²]
    Eg_ref : reference bandgap at T_ref [eV], from EQE fit at 25°C
    T_ref : reference temperature [°C], default 25

    Returns
    -------
    Eg_T : np.ndarray of bandgap values [eV], same shape as TC_arr
    """
    TC_arr = np.asarray(TC_arr, dtype=float)
    Voc_arr = np.asarray(Voc_arr, dtype=float)
    Jsc_arr = np.asarray(Jsc_arr, dtype=float)

    if not (TC_arr.shape == Voc_arr.shape == Jsc_arr.shape):
        raise ValueError(
            f"TC_arr, Voc_arr, Jsc_arr must have the same shape; "
            f"got {TC_arr.shape}, {Voc_arr.shape}, {Jsc_arr.shape}"
        )

    # η_ext = J0_rad(T_ref, Eg_ref) / J0_total(T_ref)  [dimensionless]
    # JdbFromEg returns A/cm²; J0_ref is mA/cm²
    eta_ext = JdbFromEg(T_ref, Eg_ref) * 1000.0 / J0_ref

    Eg_T = np.empty_like(TC_arr)
    bracket = (Eg_ref - 0.5, Eg_ref + 0.5)

    for i, (TC, Voc, Jsc) in enumerate(zip(TC_arr, Voc_arr, Jsc_arr)):
        kT = Vth(TC)  # thermal voltage [V]

        def f(Eg: float) -> float:
            J0_eff_mA = JdbFromEg(TC, Eg) * 1000.0 / eta_ext
            return n * kT * np.log(Jsc / J0_eff_mA) - Voc

        try:
            Eg_T[i] = brentq(f, *bracket)
        except ValueError as exc:
            raise ValueError(
                f"brentq failed at TC={TC:.1f}°C (index {i}): bracket "
                f"({bracket[0]:.3f}, {bracket[1]:.3f}) eV may not contain a root. "
                f"Voc={Voc:.4f} V, Jsc={Jsc:.4f} mA/cm²"
            ) from exc

    return Eg_T


def fit_eqe_tail_erfc(
    wavelength: np.ndarray,
    eqe: np.ndarray,
    Eg_ref: float,
) -> ErfcTailParams:
    """
    Fit the erfc tail model to the EQE edge region at 25°C.

    The model is:  EQE(E) = A · 0.5 · erfc((Eg - E) / (σ√2))
    Eg is fixed at Eg_ref; only (A, σ) are fitted.

    Fit region: wavelengths where λ > nm2eV/(Eg_ref + 0.5)  AND  EQE < 0.9·max(EQE).

    Parameters
    ----------
    wavelength : [nm], shape (N,)
    eqe : reference EQE at 25°C, shape (N,)
    Eg_ref : bandgap [eV] — pinned from EQE analysis (e.g. calc_Eg_Rau)

    Returns
    -------
    ErfcTailParams with Eg=Eg_ref, fitted A and sigma
    """
    wavelength = np.asarray(wavelength, dtype=float)
    eqe = np.asarray(eqe, dtype=float)

    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("wavelength must be strictly increasing")

    E = nm2eV / wavelength  # photon energy [eV]

    lam_min = nm2eV / (Eg_ref + 0.5)   # short-wavelength bound of fit region
    mask = (wavelength > lam_min) & (eqe < 0.9 * np.nanmax(eqe))

    if mask.sum() < 5:
        raise ValueError(
            f"Only {mask.sum()} points in the tail fit region. "
            "Check Eg_ref or extend the wavelength range."
        )

    E_tail = E[mask]
    eqe_tail = eqe[mask]

    # At lam_Eg = nm2eV/Eg_ref, the erfc model evaluates to A·0.5·erfc(0) = A/2,
    # so multiply the interpolated EQE by 2 to approximate the plateau amplitude A.
    lam_Eg = nm2eV / Eg_ref
    A_init = 2.0 * float(np.interp(lam_Eg, wavelength, eqe))
    if A_init < 1e-6:
        A_init = float(np.nanmax(eqe))

    def residuals(params: np.ndarray) -> np.ndarray:
        A, sigma = params
        return A * 0.5 * erfc((Eg_ref - E_tail) / (sigma * np.sqrt(2))) - eqe_tail

    result = least_squares(
        residuals,
        x0=[A_init, 0.04],
        bounds=([0.0, 1e-4], [1.5, 0.5]),
        method='trf',
    )
    A_fit, sigma_fit = result.x
    return ErfcTailParams(Eg=Eg_ref, A=float(A_fit), sigma=float(sigma_fit))


def fit_eqe_tail_oscillator(
    wavelength: np.ndarray,
    eqe: np.ndarray,
    Eg_ref: float,
    model: str = 'tauc_lorentz',
) -> OscillatorParams:
    """
    Fit a Tauc-Lorentz or Elliott oscillator to the EQE edge at 25°C.

    Tauc-Lorentz (model='tauc_lorentz'):
        EQE(E) = A·Γ·(E−Eg)² / ((E²−E₀²)²+Γ²·E²)  for E > Eg, else 0
        Free params: (Eg, A, E0, Gamma). Eg soft-constrained near Eg_ref.

    Elliott (model='elliott'):
        EQE(E) = A2·Γ/((E−(Eg−Eb))²+Γ²) + A·0.5·erfc((Eg−E)/(Γ√2))
        Free params: (Eg, A, Eb, Gamma, A2). OscillatorParams.E0 stores Eb.

    Parameters
    ----------
    wavelength : [nm], shape (N,)
    eqe : reference EQE at 25°C, shape (N,)
    Eg_ref : reference bandgap [eV], used as soft constraint centre
    model : 'tauc_lorentz' | 'elliott'

    Returns
    -------
    OscillatorParams
    """
    if model not in ('tauc_lorentz', 'elliott'):
        raise ValueError(f"model must be 'tauc_lorentz' or 'elliott', got '{model}'")

    wavelength = np.asarray(wavelength, dtype=float)
    eqe = np.asarray(eqe, dtype=float)

    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("wavelength must be strictly increasing")

    E = nm2eV / wavelength

    # Tail fit region
    lam_min = nm2eV / (Eg_ref + 0.5)
    mask = (wavelength > lam_min) & (eqe < 0.9 * np.nanmax(eqe))
    if mask.sum() < 5:
        raise ValueError(
            f"Only {mask.sum()} points in tail fit region. Check Eg_ref or wavelength range."
        )
    E_tail = E[mask]
    eqe_tail = eqe[mask]

    Eg_bounds = (Eg_ref - 0.1, Eg_ref + 0.1)

    if model == 'tauc_lorentz':
        # params: [Eg, A, E0, Gamma]
        # E0 lower bound = Eg_bounds[1] ensures E0 >= max(Eg_fit), keeping E0 above the bandgap.
        # A is scaled to reproduce EQE values in the tail region (not the unphysical TL peak at E0).
        x0 = [Eg_ref, np.nanmax(eqe) * 5.0, max(Eg_ref * 3.0, 3.0), 2.0]
        lo = [Eg_bounds[0], 1e-6, Eg_bounds[1], 0.01]
        hi = [Eg_bounds[1], 1e4,  10.0,         10.0]

        def residuals(p: np.ndarray) -> np.ndarray:
            Eg, A, E0, Gamma = p
            model_eqe = _tauc_lorentz(E_tail, Eg, A, E0, Gamma)
            # normalise to EQE scale so amplitude doesn't dominate
            scale = model_eqe.max() if model_eqe.max() > 0 else 1.0
            return (model_eqe / scale - eqe_tail / eqe_tail.max())

        res = least_squares(residuals, x0=x0, bounds=(lo, hi), method='trf')
        Eg_fit, A_fit, E0_fit, Gamma_fit = res.x
        # Rescale A to match EQE amplitude
        model_peak = _tauc_lorentz(E_tail, Eg_fit, A_fit, E0_fit, Gamma_fit).max()
        if model_peak > 0:
            A_fit = A_fit * eqe_tail.max() / model_peak

        return OscillatorParams(
            model='tauc_lorentz',
            Eg=float(Eg_fit),
            A=float(A_fit),
            E0=float(E0_fit),
            Gamma=float(Gamma_fit),
        )

    else:  # elliott
        # params: [Eg, A, Eb, Gamma, A2]
        x0 = [Eg_ref, 0.7 * np.nanmax(eqe), 0.01, 0.03, 0.15 * np.nanmax(eqe)]
        lo = [Eg_bounds[0], 0.0, 1e-4, 1e-3, 0.0]
        hi = [Eg_bounds[1], 2.0, 0.2,  0.5,  1.0]

        def residuals(p: np.ndarray) -> np.ndarray:
            Eg, A, Eb, Gamma, A2 = p
            return _elliott(E_tail, Eg, A, Eb, Gamma, A2) - eqe_tail

        res = least_squares(residuals, x0=x0, bounds=(lo, hi), method='trf')
        Eg_fit, A_fit, Eb_fit, Gamma_fit, A2_fit = res.x

        return OscillatorParams(
            model='elliott',
            Eg=float(Eg_fit),
            A=float(A_fit),
            E0=float(Eb_fit),  # E0 stores Eb for Elliott
            Gamma=float(Gamma_fit),
            A2=float(A2_fit),
        )


def shift_eqe_erfc(
    wavelength: np.ndarray,
    eqe_ref: np.ndarray,
    tail_params: ErfcTailParams,
    Eg_T: np.ndarray,
    TC_arr: np.ndarray,
    sigma_model: str = 'kT',
    T_ref: float = 25.0,
) -> np.ndarray:
    """
    Reconstruct EQE(T) by shifting the erfc tail to each Eg(T).

    The plateau region (λ < λ_cut) is preserved unchanged from eqe_ref.
    The tail region (λ > λ_cut) is replaced by the erfc model.

    Parameters
    ----------
    wavelength : [nm], shape (N,)
    eqe_ref : reference EQE at T_ref, shape (N,)
    tail_params : ErfcTailParams from fit_eqe_tail_erfc
    Eg_T : bandgap at each temperature [eV], shape (M,) — from extract_eg_from_voc
    TC_arr : temperatures [°C], shape (M,)
    sigma_model : 'fixed' | 'kT'
        'fixed' — σ constant at tail_params.sigma
        'kT'    — σ(T) = σ_25 · TK(T) / TK(T_ref)
    T_ref : reference temperature [°C], default 25

    Returns
    -------
    np.ndarray of shape (N, M) — EQE at each (wavelength, temperature)

    Notes
    -----
    A small discontinuity in EQE at lam_cut is expected by design: the plateau
    side carries the 25 °C reference value while the tail side carries the
    shifted-Eg model. The crossover at Eg + 2σ places the model at 97.7 % of
    A, so the step is at most ~2.3 % of the plateau amplitude.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    eqe_ref = np.asarray(eqe_ref, dtype=float)
    Eg_T = np.asarray(Eg_T, dtype=float)
    TC_arr = np.asarray(TC_arr, dtype=float)

    if len(Eg_T) != len(TC_arr):
        raise ValueError(
            f"Eg_T and TC_arr must have the same length; "
            f"got {len(Eg_T)} and {len(TC_arr)}"
        )
    if sigma_model not in ('fixed', 'kT'):
        raise ValueError(
            f"sigma_model must be 'fixed' or 'kT', got {sigma_model!r}"
        )

    E = nm2eV / wavelength  # photon energy [eV], shape (N,)
    sigma_25 = tail_params.sigma
    TK_ref = TK(T_ref)

    N = len(wavelength)
    M = len(Eg_T)
    result = np.empty((N, M), dtype=float)

    for i, (Eg, TC) in enumerate(zip(Eg_T, TC_arr)):
        if sigma_model == 'kT':
            sigma = sigma_25 * TK(TC) / TK_ref
        else:
            sigma = sigma_25

        lam_cut = nm2eV / (Eg + 2.0 * sigma)
        tail_mask = wavelength > lam_cut

        col = eqe_ref.copy()
        col[tail_mask] = tail_params.A * 0.5 * erfc(
            (Eg - E[tail_mask]) / (sigma * np.sqrt(2))
        )
        result[:, i] = col

    return result


def shift_eqe_oscillator(
    wavelength: np.ndarray,
    eqe_ref: np.ndarray,
    osc_params: OscillatorParams,
    Eg_T: np.ndarray,
    TC_arr: np.ndarray,
    Gamma_model: str = 'phonon',
    omega_ph_eV: float = 0.026,
    T_ref: float = 25.0,
) -> np.ndarray:
    """
    Reconstruct EQE(T) by shifting the oscillator tail to each Eg(T).

    The plateau region (λ < λ_cut) is preserved unchanged from eqe_ref.
    The tail region (λ > λ_cut) is replaced by the oscillator model at (Eg(T), Γ(T)).

    Parameters
    ----------
    wavelength : [nm], shape (N,)
    eqe_ref : reference EQE at T_ref, shape (N,)
    osc_params : OscillatorParams from fit_eqe_tail_oscillator
    Eg_T : bandgap at each temperature [eV], shape (M,) — from extract_eg_from_voc
    TC_arr : temperatures [°C], shape (M,)
    Gamma_model : 'fixed' | 'phonon'
        'fixed'  — Γ constant at osc_params.Gamma
        'phonon' — Γ(T) = Γ_0 + Γ_ph/(exp(ħω_ph/kT)−1), anchored at T_ref.
                   Requires osc_params.Gamma_ph > 0; if zero, falls back to 'fixed'.
    omega_ph_eV : phonon energy [eV]; default 0.026 (Si ~26 meV); use 0.015 for perovskite
    T_ref : reference temperature [°C]

    Returns
    -------
    np.ndarray of shape (N, M)

    Notes
    -----
    For Tauc-Lorentz, λ_cut = nm2eV/Eg (hard bandgap edge; TL is identically zero below Eg).
    For Elliott, λ_cut = nm2eV/(Eg + 2·Gamma) (same as erfc; sub-gap tail from erfc term).
    A small discontinuity at λ_cut is expected by design.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    eqe_ref = np.asarray(eqe_ref, dtype=float)
    Eg_T = np.asarray(Eg_T, dtype=float)
    TC_arr = np.asarray(TC_arr, dtype=float)

    if len(Eg_T) != len(TC_arr):
        raise ValueError(
            f"Eg_T and TC_arr must have the same length; "
            f"got {len(Eg_T)} and {len(TC_arr)}"
        )
    if Gamma_model not in ('fixed', 'phonon'):
        raise ValueError(
            f"Gamma_model must be 'fixed' or 'phonon', got {Gamma_model!r}"
        )

    E = nm2eV / wavelength
    Gamma_25 = osc_params.Gamma
    Gamma_ph = osc_params.Gamma_ph
    kT_ref = Vth(T_ref)  # eV

    # Pre-compute Γ_0 for phonon model (anchors Γ(T_ref) = Gamma_25)
    if Gamma_model == 'phonon' and Gamma_ph > 0.0:
        phonon_ref = np.expm1(omega_ph_eV / kT_ref)   # exp(ħω/kT_ref) - 1
        Gamma_0 = Gamma_25 - Gamma_ph / phonon_ref
    else:
        Gamma_0 = Gamma_25   # unused in 'fixed' mode

    N = len(wavelength)
    M = len(Eg_T)
    result = np.empty((N, M), dtype=float)

    for i, (Eg, TC) in enumerate(zip(Eg_T, TC_arr)):
        if Gamma_model == 'phonon' and Gamma_ph > 0.0:
            kT = Vth(TC)
            Gamma = Gamma_0 + Gamma_ph / np.expm1(omega_ph_eV / kT)
        else:
            Gamma = Gamma_25

        # Tauc-Lorentz is identically zero below Eg — cut at the hard bandgap edge.
        # Elliott has sub-gap tail from its erfc term — cut 2·Gamma above Eg (same as erfc).
        if osc_params.model == 'tauc_lorentz':
            lam_cut = nm2eV / Eg
        else:
            lam_cut = nm2eV / (Eg + 2.0 * Gamma)
        tail_mask = wavelength > lam_cut

        # Start with eqe_ref, then replace the tail region
        col = eqe_ref.copy()

        # Compute oscillator model for tail region
        E_tail = E[tail_mask]
        if osc_params.model == 'tauc_lorentz':
            col[tail_mask] = _tauc_lorentz(E_tail, Eg, osc_params.A, osc_params.E0, Gamma)
        else:  # elliott
            col[tail_mask] = _elliott(E_tail, Eg, osc_params.A, osc_params.E0, Gamma, osc_params.A2)

        result[:, i] = col

    return result


def fit_gamma_phonon(
    wavelength: np.ndarray,
    eqe_ref: np.ndarray,
    osc_params: OscillatorParams,
    Eg_T: np.ndarray,
    TC_arr: np.ndarray,
    Jsc_measured_T: np.ndarray,
    spectra: np.ndarray,
    xspec: np.ndarray,
    omega_ph_eV: float = 0.026,
    T_ref: float = 25.0,
) -> OscillatorParams:
    """
    Fit the phonon broadening coefficient Gamma_ph to measured Jsc(T).

    Minimises sum((Jsc_model(T) - Jsc_measured_T)²) over Gamma_ph by
    integrating the oscillator-shifted EQE against the provided spectra.

    Parameters
    ----------
    wavelength : [nm], shape (N,)
    eqe_ref : reference EQE at T_ref, shape (N,)
    osc_params : OscillatorParams — Gamma_ph will be updated in the returned copy
    Eg_T : bandgap array [eV], shape (M,)
    TC_arr : temperatures [°C], shape (M,)
    Jsc_measured_T : measured Jsc at each temperature [any consistent units], shape (M,)
    spectra : spectra array, shape (N, M) — one spectrum per temperature, on xspec grid
    xspec : wavelength axis for spectra [nm], shape (N,)
    omega_ph_eV : phonon energy [eV]; default 0.026 (~26 meV, Si)
    T_ref : reference temperature [°C]

    Returns
    -------
    OscillatorParams with Gamma_ph set to the fitted value
    """
    from copy import copy
    from scipy.interpolate import interp1d

    wavelength = np.asarray(wavelength, dtype=float)
    eqe_ref = np.asarray(eqe_ref, dtype=float)
    Eg_T = np.asarray(Eg_T, dtype=float)
    TC_arr = np.asarray(TC_arr, dtype=float)
    Jsc_measured_T = np.asarray(Jsc_measured_T, dtype=float)
    spectra = np.asarray(spectra, dtype=float)
    xspec = np.asarray(xspec, dtype=float)

    # Interpolate spectra onto the EQE wavelength grid once
    spec_interp = interp1d(
        xspec, spectra, axis=0, bounds_error=False, fill_value=0.0
    )(wavelength)

    def _jsc_model(Gamma_ph: float) -> np.ndarray:
        osc_temp = copy(osc_params)
        osc_temp.Gamma_ph = Gamma_ph
        eqe_T = shift_eqe_oscillator(
            wavelength, eqe_ref, osc_temp, Eg_T, TC_arr,
            Gamma_model='phonon', omega_ph_eV=omega_ph_eV, T_ref=T_ref,
        )
        # Jsc ∝ ∫ EQE(λ) · spectrum(λ) · λ dλ
        integrand = eqe_T * spec_interp * wavelength[:, np.newaxis]
        return np.trapezoid(integrand, wavelength, axis=0)

    # Normalise so residuals are scale-independent
    Jsc_mean = Jsc_measured_T.mean()
    Jsc_norm = Jsc_measured_T / (Jsc_mean + 1e-30)

    def residuals(p: np.ndarray) -> np.ndarray:
        Gamma_ph = p[0]
        jsc_pred = _jsc_model(Gamma_ph)
        jsc_pred_norm = jsc_pred / (jsc_pred.mean() + 1e-30)
        return jsc_pred_norm - Jsc_norm

    res = least_squares(
        residuals,
        x0=[max(osc_params.Gamma_ph, 0.01)],
        bounds=([0.0], [osc_params.Gamma * 2.0 + 0.5]),
        method='trf',
    )
    result = copy(osc_params)
    result.Gamma_ph = float(res.x[0])
    return result
