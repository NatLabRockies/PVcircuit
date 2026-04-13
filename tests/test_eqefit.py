"""
Test suite for pvcircuit.eqefit module.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
import pvcircuit as pvc
import pvcircuit.eqefit as eqefit
from pvcircuit.eqefit import (
    ErfcTailParams,
    OscillatorParams,
    extract_eg_from_voc,
    fit_eqe_tail_erfc,
    fit_eqe_tail_oscillator,
    shift_eqe_erfc,
    shift_eqe_oscillator,
    fit_gamma_phonon,
    _tauc_lorentz,
    _elliott,
)
from pvcircuit.qe import JdbFromEg, nm2eV
from pvcircuit.conversions import TK, Vth
from scipy.special import erfc


def test_import():
    """Test that eqefit module is accessible from pvc package."""
    assert hasattr(pvc, 'eqefit')
    assert hasattr(eqefit, 'extract_eg_from_voc')


def test_extract_eg_round_trip():
    """Synthesise Voc(T) from known Eg(T), recover Eg(T), check error < 1 meV."""
    TC_arr = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    dEg_dT = -3e-4   # eV/°C  (representative value)
    Eg_ref = 1.4
    Eg_true = Eg_ref + dEg_dT * (TC_arr - 25.0)

    n = 1.0
    T_ref = 25.0
    Jsc_mA = 20.0   # mA/cm², held constant for simplicity

    # J0_ref = J0_rad at T_ref (ideal radiative limit → η_ext = 1)
    J0_ref = JdbFromEg(T_ref, Eg_ref) * 1000.0  # A/cm² → mA/cm²

    # Synthesise Voc(T): Voc = n·kT·ln(Jsc / J0_eff)
    # With η_ext=1: J0_eff = JdbFromEg(T, Eg_true)*1000
    Voc_arr = np.array([
        n * Vth(T) * np.log(Jsc_mA / (JdbFromEg(T, Eg) * 1000.0))
        for T, Eg in zip(TC_arr, Eg_true)
    ])
    Jsc_arr = np.full_like(TC_arr, Jsc_mA)

    Eg_recovered = extract_eg_from_voc(TC_arr, Voc_arr, Jsc_arr, n, J0_ref, Eg_ref, T_ref)

    np.testing.assert_allclose(Eg_recovered, Eg_true, atol=1e-3)


def test_fit_eqe_tail_erfc_recovers_params():
    """Fit erfc model to synthetic tail, verify A and sigma within 1%."""
    wavelength = np.linspace(800, 1400, 600)
    E = nm2eV / wavelength
    Eg_true = 1.0
    A_true = 0.85
    sigma_true = 0.04

    eqe = A_true * 0.5 * erfc((Eg_true - E) / (sigma_true * np.sqrt(2)))

    params = fit_eqe_tail_erfc(wavelength, eqe, Eg_ref=Eg_true)

    assert isinstance(params, ErfcTailParams)
    assert params.Eg == Eg_true
    assert abs(params.A - A_true) / A_true < 0.01, f"A error: {abs(params.A - A_true)/A_true:.3%}"
    assert abs(params.sigma - sigma_true) / sigma_true < 0.01, f"sigma error: {abs(params.sigma - sigma_true)/sigma_true:.3%}"


def test_shift_eqe_erfc_plateau_preserved():
    """EQE below the cutoff wavelength must be identical to eqe_ref."""
    wavelength = np.linspace(400, 1400, 1000)
    eqe_ref = np.ones_like(wavelength) * 0.85

    tail_params = ErfcTailParams(Eg=1.1, A=0.85, sigma=0.04)
    Eg_T = np.array([1.08, 1.10, 1.12])
    TC_arr = np.array([75.0, 25.0, -25.0])   # higher T → lower Eg

    result = shift_eqe_erfc(wavelength, eqe_ref, tail_params, Eg_T, TC_arr, sigma_model='fixed')

    assert result.shape == (len(wavelength), 3)
    for i, Eg in enumerate(Eg_T):
        cutoff_nm = nm2eV / (Eg + 2 * tail_params.sigma)
        plateau_mask = wavelength < cutoff_nm
        np.testing.assert_array_almost_equal(
            result[plateau_mask, i], eqe_ref[plateau_mask],
            err_msg=f"Plateau changed at T index {i}"
        )


def test_shift_eqe_erfc_plateau_preserved_kt():
    """Plateau preservation must also hold for sigma_model='kT' using the kT-adjusted cutoff."""
    wavelength = np.linspace(400, 1400, 1000)
    eqe_ref = np.ones_like(wavelength) * 0.85

    tail_params = ErfcTailParams(Eg=1.1, A=0.85, sigma=0.04)
    Eg_T = np.array([1.08, 1.10, 1.12])
    TC_arr = np.array([75.0, 25.0, -25.0])

    result = shift_eqe_erfc(wavelength, eqe_ref, tail_params, Eg_T, TC_arr, sigma_model='kT')

    assert result.shape == (len(wavelength), 3)
    for i, (Eg, TC) in enumerate(zip(Eg_T, TC_arr)):
        sigma_kT = tail_params.sigma * TK(TC) / TK(25.0)
        cutoff_nm = nm2eV / (Eg + 2 * sigma_kT)
        plateau_mask = wavelength < cutoff_nm
        np.testing.assert_array_almost_equal(
            result[plateau_mask, i], eqe_ref[plateau_mask],
            err_msg=f"Plateau changed at T index {i} with kT sigma"
        )


def test_shift_eqe_erfc_jsc_increases_with_T():
    """Decreasing Eg (higher T) must give higher integrated EQE × λ."""
    wavelength = np.linspace(400, 1400, 1000)
    E = nm2eV / wavelength
    Eg_25 = 1.1
    A, sigma = 0.85, 0.04
    eqe_ref = A * 0.5 * erfc((Eg_25 - E) / (sigma * np.sqrt(2)))

    tail_params = ErfcTailParams(Eg=Eg_25, A=A, sigma=sigma)
    # Three temperatures: cold, reference, hot
    TC_arr = np.array([-25.0, 25.0, 75.0])
    Eg_T = np.array([1.13, 1.10, 1.07])   # Eg decreases with T

    result = shift_eqe_erfc(wavelength, eqe_ref, tail_params, Eg_T, TC_arr)

    # Proxy for Jsc: integrate EQE · λ (proportional to photon count)
    jsc_proxy = np.trapezoid(result * wavelength[:, np.newaxis], wavelength, axis=0)
    assert jsc_proxy[0] < jsc_proxy[1] < jsc_proxy[2], (
        f"Expected monotone increase, got {jsc_proxy}"
    )


def test_fit_eqe_tail_oscillator_tauc_lorentz():
    """Fit Tauc-Lorentz to synthetic data, recover Eg within 5 meV."""
    wavelength = np.linspace(600, 1300, 700)
    E = nm2eV / wavelength
    Eg_true = 1.12
    E0_true = 3.4    # Si-like resonance energy
    Gamma_true = 2.0
    A_true = 8.0

    mask_above = E > Eg_true
    eqe = np.where(
        mask_above,
        A_true * Gamma_true * (E - Eg_true)**2 / ((E**2 - E0_true**2)**2 + Gamma_true**2 * E**2),
        0.0,
    )
    # Clip to [0,1] range typical of EQE
    eqe = np.clip(eqe / eqe.max() * 0.9, 0.0, 1.0)

    params = fit_eqe_tail_oscillator(wavelength, eqe, Eg_ref=Eg_true, model='tauc_lorentz')

    assert isinstance(params, OscillatorParams)
    assert params.model == 'tauc_lorentz'
    assert abs(params.Eg - Eg_true) < 0.005, f"Eg error: {abs(params.Eg - Eg_true)*1000:.1f} meV"


def test_fit_eqe_tail_oscillator_elliott():
    """Fit Elliott model to synthetic exciton + continuum, recover Eg within 10 meV."""
    wavelength = np.linspace(600, 1100, 500)
    E = nm2eV / wavelength
    Eg_true = 1.6    # perovskite-like
    Eb_true = 0.01   # exciton binding energy [eV]
    Gamma_true = 0.03
    A_cont_true = 0.80
    A_exc_true = 0.15

    # Exciton peak at Eg - Eb
    eqe_exc = A_exc_true * Gamma_true / ((E - (Eg_true - Eb_true))**2 + Gamma_true**2)
    eqe_cont = A_cont_true * 0.5 * erfc((Eg_true - E) / (Gamma_true * np.sqrt(2)))
    eqe = np.clip(eqe_exc + eqe_cont, 0.0, 1.0)

    params = fit_eqe_tail_oscillator(wavelength, eqe, Eg_ref=Eg_true, model='elliott')

    assert params.model == 'elliott'
    assert abs(params.Eg - Eg_true) < 0.010, f"Eg error: {abs(params.Eg - Eg_true)*1000:.1f} meV"


def test_shift_eqe_oscillator_plateau_preserved():
    """EQE below cutoff wavelength must equal eqe_ref."""
    wavelength = np.linspace(400, 1400, 1000)
    E = nm2eV / wavelength
    Eg_25 = 1.12
    eqe_ref = np.where(E > Eg_25, 0.85, 0.0)   # idealised square EQE

    osc_params = OscillatorParams(
        model='tauc_lorentz', Eg=Eg_25, A=100.0, E0=3.4, Gamma=2.0
    )
    Eg_T = np.array([1.14, 1.12, 1.10])
    TC_arr = np.array([-25.0, 25.0, 75.0])

    result = shift_eqe_oscillator(
        wavelength, eqe_ref, osc_params, Eg_T, TC_arr, Gamma_model='fixed'
    )

    assert result.shape == (len(wavelength), 3)
    for i, Eg in enumerate(Eg_T):
        cutoff_nm = nm2eV / Eg   # TL: hard bandgap edge
        plateau_mask = wavelength < cutoff_nm
        np.testing.assert_array_almost_equal(
            result[plateau_mask, i], eqe_ref[plateau_mask],
            err_msg=f"Plateau changed at temperature index {i}"
        )


def test_shift_eqe_oscillator_jsc_increases_with_T():
    """Decreasing Eg (higher T) must increase integrated EQE·λ.

    Uses a flat step EQE below all Eg_T values so the plateau region expands
    as Eg decreases, producing a strictly monotone Jsc proxy.  (A purely
    Tauc-Lorentz eqe_ref would be identically zero at and below the bandgap
    edge, making the integral insensitive to Eg shifts.)
    """
    wavelength = np.linspace(600, 1400, 800)
    E = nm2eV / wavelength
    Eg_25 = 1.12

    # Flat EQE at 0.85 above 0.95 eV (< all Eg_T values) — zero below 0.95 eV.
    # As Eg decreases, lam_cut = nm2eV/Eg increases and the plateau grows.
    eqe_ref = np.where(E > 0.95, 0.85, 0.0)

    osc_params = OscillatorParams(
        model='tauc_lorentz', Eg=Eg_25, A=100.0, E0=3.4, Gamma=2.0
    )
    TC_arr = np.array([-25.0, 25.0, 75.0])
    Eg_T = np.array([1.15, 1.12, 1.09])   # decreases with T

    result = shift_eqe_oscillator(
        wavelength, eqe_ref, osc_params, Eg_T, TC_arr, Gamma_model='fixed'
    )

    jsc_proxy = np.trapezoid(result * wavelength[:, np.newaxis], wavelength, axis=0)
    assert jsc_proxy[0] < jsc_proxy[1] < jsc_proxy[2], (
        f"Expected monotone Jsc increase, got {jsc_proxy}"
    )


def test_fit_gamma_phonon_recovers_Gamma_ph():
    """Synthesise Jsc(T) with known Gamma_ph (Elliott) and recover it within 20%.

    Uses the Elliott model because its erfc sub-gap tail is the tail region that
    shift_eqe_oscillator replaces — making Jsc genuinely sensitive to Gamma(T).
    For Tauc-Lorentz the tail region has E < Eg, so TL ≡ 0 there regardless of
    Gamma and the optimizer has no gradient to follow.
    """
    wavelength = np.linspace(500, 1100, 1200)
    E = nm2eV / wavelength
    Eg_25 = 1.60   # perovskite-like bandgap [eV]
    Gamma_ref = 0.030  # 30 meV broadening at 25°C
    Gamma_ph_true = 0.012  # 12 meV phonon coupling — changes Gamma ~15 % over ±50 °C

    osc_params = OscillatorParams(
        model='elliott', Eg=Eg_25, A=0.85, E0=0.010, Gamma=Gamma_ref, A2=0.12, Gamma_ph=0.0
    )
    # Reference EQE: Elliott model evaluated at 25°C
    eqe_ref = np.clip(_elliott(E, Eg_25, 0.85, 0.010, Gamma_ref, 0.12), 0.0, 1.0)

    TC_arr = np.array([-25.0, 0.0, 25.0, 50.0, 75.0])
    Eg_T = Eg_25 + (-4e-4) * (TC_arr - 25.0)   # typical perovskite temperature coefficient

    # Synthesise Jsc(T) with the true phonon coupling (phonon energy 15 meV for perovskite)
    osc_true = OscillatorParams(
        model='elliott', Eg=Eg_25, A=0.85, E0=0.010, Gamma=Gamma_ref, A2=0.12,
        Gamma_ph=Gamma_ph_true
    )
    eqe_T_true = shift_eqe_oscillator(
        wavelength, eqe_ref, osc_true, Eg_T, TC_arr,
        Gamma_model='phonon', omega_ph_eV=0.015,
    )

    spectra = np.ones((len(wavelength), len(TC_arr)))  # flat AM1.5 proxy
    Jsc_measured_T = np.trapezoid(
        eqe_T_true * wavelength[:, np.newaxis], wavelength, axis=0
    )

    fitted = fit_gamma_phonon(
        wavelength, eqe_ref, osc_params, Eg_T, TC_arr,
        Jsc_measured_T, spectra, xspec=wavelength,
        omega_ph_eV=0.015,
    )

    assert isinstance(fitted, OscillatorParams)
    assert abs(fitted.Gamma_ph - Gamma_ph_true) / Gamma_ph_true < 0.20, (
        f"Gamma_ph error: {abs(fitted.Gamma_ph - Gamma_ph_true)/Gamma_ph_true:.1%}"
    )
