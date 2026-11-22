"""
This is the PVcircuit Package.
    pvcircuit.qe    # functions for QE analysis
"""

from __future__ import annotations

import math  # simple math
import warnings
from collections.abc import Callable, Sequence
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import matplotlib as mpl  # plotting
import matplotlib.pyplot as plt  # plotting
import numpy as np  # arrays
import pandas as pd  # dataframes
from loguru import logger
from scipy import constants  # physical constants
from scipy.integrate import trapezoid
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import (
    curve_fit,
    fsolve,
    least_squares,
)
from scipy.special import gammaincc, ndtr  # special functions

import pvcircuit as pvc
from pvcircuit import conversions as convert

# colors
junctioncolors = [
    ["black"],  # 0J
    ["red"],  # 1J
    ["blue", "red"],  # 2J
    ["blue", "green", "red"],  # 3J
    ["blue", "green", "orange", "red"],  # 4J
    ["purple", "blue", "green", "orange", "red"],  # 5J
    ["purple", "blue", "green", "black", "orange", "red"],
]  # 6J

# constants
k_q = constants.k / constants.e
hc_k = constants.h * constants.c / constants.k * 1e9  # [nm*K] used in radiative integrand exp(hc/\lambdakT) when \lambda is in nm
DB_PREFIX = 2.0 * np.pi * constants.e * (constants.k / constants.h) ** 3 / (constants.c) ** 2 / 1.0e4  # [A/(cm^2*K^3)] \approx 1.0133e-8, used for Jdb [A/cm^2]
nm2eV = constants.h * constants.c / constants.e * 1e9  # [eV*nm] \approx 1239.84, converts photon energy [eV] <-> wavelength [nm]
# JCONST converts trapezoid(EQE*Pspec*\lambda, d\lambda) into a current density. Units cancel
# as follows: integrand [W/m^2/nm * nm * nm -> W*nm/m^2] \times JCONST = mA/cm^2.
#   1000  : A -> mA
#   1/100 : 1/m -> 1/cm        (first /100)
#   1/100 : 1/m -> 1/cm        (second /100)
#   1/nm2eV: nm -> eV (so photon energy in eV cancels with EQE per-photon)
JCONST = 1000 / 100 / 100 / nm2eV  # mA/cm^2
DBWVL_PREFIX = 2.0 * np.pi * constants.c * constants.e / 100 / 100  # [A/cm^2] wavelength-domain Jdb prefactor


# standard spectra
ASTMfile = pvc.datapath.joinpath("ASTMG173.csv")

try:
    dfrefspec = pd.read_csv(ASTMfile, index_col=0, header=2)
except Exception as err:
    # Fail if the reference don't exist
    raise RuntimeError(f"Cannot read ASTM reference spectra from {ASTMfile} (pvcpath={pvc.pvcpath}, datapath={pvc.datapath})") from err

wvl = dfrefspec.index.to_numpy(dtype=np.float64, copy=True)
refspec = dfrefspec.to_numpy(dtype=np.float64, copy=True)  # all three reference spectra
refnames = ["space", "global", "direct"]
AM0 = refspec[:, 0]  # 280-4000 nm table integral: 1347.9 W/m^2 (full-spectrum E490 AM0 total: 1366.1 W/m^2)
AM15G = refspec[:, 1]  # 280-4000 nm table integral: 1000.4 W/m^2 (ASTM G173-03 Sec. 4.3)
AM15D = refspec[:, 2]  # 280-4000 nm table integral: 900.1 W/m^2 (ASTM G173-03 Sec. 4.3)


def ordinal(n: int) -> str:
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = suffixes.get(n % 10, "th")
    return f"{n}{suffix}"


def _eq_solve_Eg(Eg: float, *data: np.ndarray) -> float:
    x, y = data
    return trapezoid(x * y, x) / trapezoid(y, x) - Eg


def _gaussian(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def _jdb_from_eqe_columns(
    eqe_2d: np.ndarray,
    wl_1d: np.ndarray,
    TC: float,
    Eguess: float | Sequence[float] | None = 1.0,
    kTfilter: float = 3.0,
    max_iter: int = 50,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute detailed-balance Jdb [A/cm^2] and step-equivalent Eg [eV] per EQE column."""
    Vthlocal = convert.Vth(TC)
    wl = np.asarray(wl_1d, dtype=np.float64).ravel()
    eqe = np.asarray(eqe_2d, dtype=np.float64)
    if eqe.ndim == 1:
        eqe = eqe[:, np.newaxis]
    nQlams, njuncs = eqe.shape
    if wl.size != nQlams:
        raise ValueError(f"length of xEQE ({wl.size}) does not match EQE rows ({nQlams})")
    if nQlams < 2:
        raise ValueError("xEQE must contain at least 2 wavelength points")

    if not np.all(np.diff(wl) > 0):
        order = np.argsort(wl, kind="stable")
        wl = wl[order]
        eqe = eqe[order, :]
        if np.any(np.diff(wl) == 0):
            raise ValueError("xEQE contains duplicate wavelength entries")

    EkT = nm2eV / Vthlocal / wl
    with np.errstate(over="ignore", divide="ignore"):
        blackbody = DBWVL_PREFIX / (wl * 1e-9) ** 4 / np.expm1(EkT)  # shape (nQlams,)

    # Data-derived half-maximum crossing per column prevents Eguess - kTfilter*Vth
    # from cutting above a low-bandgap edge (e.g. Ge 0.67-0.74 eV).
    E_grid = nm2eV / wl
    e_init = np.empty(njuncs, dtype=np.float64)
    if Eguess is None:
        user_guesses = None
    elif np.ndim(Eguess) == 0:
        user_guesses = np.full(njuncs, float(Eguess), dtype=np.float64)
    else:
        user_guesses = np.asarray(Eguess, dtype=np.float64).ravel()
        if user_guesses.size != njuncs:
            raise ValueError(f"Eguess length ({user_guesses.size}) must match njuncs ({njuncs})")

    for j in range(njuncs):
        col = eqe[:, j]
        finite_col = np.where(np.isfinite(col), col, 0.0)
        q_max = float(np.max(finite_col))
        if q_max <= 0.0:
            raise ValueError(f"EQE column {j} has no positive values; cannot compute Jdb")
        # Longest wavelength where EQE >= 0.5 * max(EQE) locates the primary absorption edge
        above_half = np.flatnonzero(finite_col >= 0.5 * q_max)
        e_half = float(E_grid[above_half[-1]])
        if user_guesses is None or float(user_guesses[j]) == 1.0:
            e_init[j] = e_half
        else:
            # Guard against an explicit Eguess whose filter cut lies above the band edge
            ug = float(user_guesses[j])
            e_init[j] = min(ug, e_half) if (ug - kTfilter * Vthlocal) > e_half else ug

    Jdb_out = np.empty(njuncs, dtype=np.float64)
    Eg_out = np.empty(njuncs, dtype=np.float64)
    db_integrand = np.zeros((nQlams, njuncs), dtype=np.float64)

    for j in range(njuncs):
        eg_cur = e_init[j]
        eg_prev2 = np.nan
        col = np.where(np.isfinite(eqe[:, j]), eqe[:, j], 0.0)
        converged = False
        for _ in range(max_iter):
            e_cut = eg_cur - Vthlocal * kTfilter
            nmfilter = nm2eV / e_cut if e_cut > 0.0 else np.inf
            eqe_filt = np.where(wl < nmfilter, col, 0.0)
            integrand_j = blackbody * eqe_filt
            jdb_j = float(trapezoid(integrand_j, x=(wl * 1e-9)))
            if not math.isfinite(jdb_j) or jdb_j <= 0.0:
                raise ValueError(f"Filtered radiative integral for EQE column {j} is non-positive ({jdb_j}); check sub-gap EQE noise or wavelength range.")
            eg_new = EgFromJdb(TC, jdb_j)
            assert eg_new is not None
            rel_change = abs(eg_new - eg_cur) / abs(eg_new)
            if debug:
                print(j, eg_new, rel_change)
            db_integrand[:, j] = integrand_j
            Jdb_out[j] = jdb_j
            Eg_out[j] = eg_new
            if rel_change < 1e-6 or (math.isfinite(eg_prev2) and abs(eg_new - eg_prev2) == 0.0):
                converged = True
                break
            eg_prev2 = eg_cur
            eg_cur = eg_new
        if not converged:
            warnings.warn(
                f"Jdb from EQE column {j} did not converge within {max_iter} passes (last relative step {rel_change:.2e}).",
                RuntimeWarning,
                stacklevel=3,
            )

    return Jdb_out, Eg_out, db_integrand


def JdbMD(
    EQE: np.ndarray | list[float],
    xEQE: np.ndarray | list[float],
    TC: float,
    Eguess: float | Sequence[float] | None = 1.0,
    kTfilter: int = 3,
    bplot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate detailed-balance reverse saturation current density from EQE.

    Parameters
    ----------
    EQE : 1-D or 2-D array-like
        External quantum efficiency (unitless), shape (n_wvl,) or (n_wvl, njuncs).
    xEQE : 1-D array-like
        Wavelengths [nm], or a 2-element '(start, stop)' range for evenly spaced data.
    TC : float
        Temperature [degC].
    Eguess : float, sequence of float, or None, optional
        Initial bandgap guess [eV] for the sub-gap noise filter. When left at
        the default (1.0 or None), each column starts from its half-maximum
        band-edge energy.
    kTfilter : int or float, optional
        Sub-gap noise rejection offset in multiples of kT: EQE below
        'Eg - kTfilter * Vth' is zeroed on each pass (note: for strongly
        broadened tails with sigma >~ 1.5*Vth this also truncates genuine
        sub-gap tail emission).
    bplot : bool, optional
        Plot radiative emission integrands if True.

    Returns
    -------
    Jdb : np.ndarray
        Detailed-balance saturation current density in **A/cm^2**.
    Egnew : np.ndarray
        Step-equivalent bandgap in **eV** inverted from 'Jdb' via 'EgFromJdb'.
    """
    EQE_arr = np.asarray(EQE, dtype=np.float64)
    if EQE_arr.ndim == 1:
        (nQlams,) = EQE_arr.shape
        njuncs = 1
    elif EQE_arr.ndim == 2:
        nQlams, njuncs = EQE_arr.shape
    else:
        raise ValueError(f"EQE must be 1D or 2D, got ndim={EQE_arr.ndim}")

    if len(xEQE) == 2 and nQlams != 2:  # evenly spaced x-values (start, stop)
        start, stop = xEQE
        wl = np.linspace(start, stop, nQlams, dtype=np.float64)
    else:
        wl = np.asarray(xEQE, dtype=np.float64)

    if wl.ndim != 1:
        raise ValueError(f"xEQE must be 1D, got ndim={wl.ndim}")
    if len(wl) != nQlams:
        raise ValueError(f"length of xEQE ({len(wl)}) does not match EQE rows ({nQlams})")

    Jdb_out, Egnew, DBintegral = _jdb_from_eqe_columns(EQE_arr, wl, TC, Eguess=Eguess, kTfilter=kTfilter, debug=bplot)

    if bplot:
        wl_sorted = np.sort(wl)
        _, eax = plt.subplots()
        eax.plot(wl_sorted, DBintegral[:, 0], c="blue", lw=2, marker=".")
        if njuncs > 1:
            reax = eax.twinx()
            reax.plot(wl_sorted, DBintegral[:, 1], c="red", lw=2, marker=".")

    return Jdb_out, Egnew


def PintMD(Pspec: str | np.ndarray, xspec: np.ndarray = wvl) -> np.ndarray:
    # optical power of spectrum over full range
    return JintMD(None, None, Pspec, xspec)


def JintMD(EQE: np.ndarray | float | None, xEQE: np.ndarray | Sequence[float] | None, Pspec: str | np.ndarray, xspec: np.ndarray = wvl) -> np.ndarray:
    """Integrate spectral power [W/m^2] or short-circuit current density [mA/cm^2].

    If 'EQE' is None: integrates optical power 'int Pspec dlambda' in [W/m^2]
    over the overlap with 'xEQE' (full 'xspec' if 'xEQE' is None).
    Otherwise: integrates 'Jsc = JCONST * int Pspec * EQE * lambda dlambda' in
    **mA/cm^2** on the union of spectral and EQE wavelength nodes inside the
    overlap interval '[lo, hi]', with exact linear interpolation at off-grid
    endpoints.

    Parameters
    ----------
    EQE : array-like, float, or None
        External quantum efficiency (scalar for step EQE over 'xEQE', 1-D, 2-D,
        or None for broadband power).
    xEQE : array-like, 2-tuple '(start, stop)', or None
        Wavelengths [nm] for 'EQE', '(start, stop)' bounds [nm] for evenly
        spaced or step EQE, or None to use 'xspec'.
    Pspec : str or array-like
        Reference spectrum key ('space', 'global', 'direct', or '' for all
        three) or spectral irradiance array [W/m^2/nm].
    xspec : array-like, optional
        Wavelength grid [nm] for 'Pspec' (defaults to ASTM 'wvl').

    .. note::
        **Return unit**: when 'EQE' is supplied the result is in **mA/cm^2**.
        'Junction.Jext' is stored in **A/cm^2**, so divide by 1000 before
        assigning to a junction.
    """
    # check spectra input
    if isinstance(Pspec, str):
        if Pspec in dfrefspec.columns:
            Pspec_arr = dfrefspec[Pspec].to_numpy(dtype=np.float64, copy=True)
        else:
            Pspec_arr = dfrefspec.to_numpy(dtype=np.float64, copy=True)
    else:
        Pspec_arr = np.array(Pspec, dtype=np.float64)

    pspec_was_1d = Pspec_arr.ndim == 1
    if pspec_was_1d:
        (nSlams,) = Pspec_arr.shape
        nspecs = 1
        Pspec_2d = Pspec_arr[:, np.newaxis]
    elif Pspec_arr.ndim == 2:
        nSlams, nspecs = Pspec_arr.shape
        Pspec_2d = Pspec_arr
    else:
        raise ValueError(f"Pspec must be 1D or 2D, got ndim={Pspec_arr.ndim}")

    xspec_arr = np.asarray(xspec, dtype=np.float64)
    if xspec_arr.ndim != 1:
        raise ValueError(f"xspec must be 1D, got ndim={xspec_arr.ndim}")
    if len(xspec_arr) != nSlams:
        raise ValueError(f"length of xspec ({len(xspec_arr)}) does not match Pspec rows ({nSlams})")
    if nSlams > 1 and not np.all(np.diff(xspec_arr) > 0):
        s_order = np.argsort(xspec_arr, kind="stable")
        xspec_arr = xspec_arr[s_order]
        Pspec_2d = Pspec_2d[s_order, :]

    # check EQE input
    if EQE is None or (np.ndim(EQE) == 0 and not bool(np.any(EQE)) and isinstance(EQE, (bool, np.bool_))):
        nQlams = 1
        njuncs = 0
        EQE_arr = np.zeros((1, 1), dtype=np.float64)
    else:
        EQE_raw = np.asarray(EQE)
        if EQE_raw.ndim == 0:
            if EQE_raw.item() is None or (isinstance(EQE_raw.item(), (bool, np.bool_)) and not EQE_raw.item()):
                nQlams = 1
                njuncs = 0
                EQE_arr = np.zeros((1, 1), dtype=np.float64)
            else:
                nQlams = 1
                njuncs = 1
                EQE_arr = np.array([[float(EQE_raw)]], dtype=np.float64)
        elif EQE_raw.ndim == 1:
            (nQlams,) = EQE_raw.shape
            njuncs = 1
            EQE_arr = np.asarray(EQE_raw, dtype=np.float64)[:, np.newaxis]
        elif EQE_raw.ndim == 2:
            nQlams, njuncs = EQE_raw.shape
            EQE_arr = np.asarray(EQE_raw, dtype=np.float64)
        else:
            raise ValueError(f"EQE must be scalar, 1D or 2D, got ndim={EQE_raw.ndim}")

    # check xEQE range input
    if xEQE is None:
        xEQE_arr = xspec_arr.copy()
        start, stop = float(xspec_arr[0]), float(xspec_arr[-1])
    else:
        xEQE_raw = np.asarray(xEQE, dtype=np.float64)
        if xEQE_raw.ndim == 0:
            xEQE_arr = xspec_arr.copy()
            start, stop = float(xspec_arr[0]), float(xspec_arr[-1])
        elif xEQE_raw.ndim == 1:
            nxEQE = xEQE_raw.size
            start, stop = float(np.min(xEQE_raw)), float(np.max(xEQE_raw))
            if nxEQE == 2 and nQlams != 2:
                xEQE_arr = np.linspace(start, stop, max(nQlams, 2), dtype=np.float64)
            else:
                xEQE_arr = xEQE_raw
        else:
            raise ValueError(f"xEQE must be 1D, got ndim={xEQE_raw.ndim}")

    if nQlams == 1 and njuncs > 0:
        EQE_arr = np.full((xEQE_arr.size, 1), float(EQE_arr[0, 0]), dtype=np.float64)
    elif njuncs > 0 and len(xEQE_arr) != nQlams:
        raise ValueError(f"length of xEQE ({len(xEQE_arr)}) does not match EQE rows ({nQlams})")

    if njuncs > 0 and xEQE_arr.size > 1 and not np.all(np.diff(xEQE_arr) > 0):
        q_order = np.argsort(xEQE_arr, kind="stable")
        xEQE_arr = xEQE_arr[q_order]
        EQE_arr = EQE_arr[q_order, :]

    lo = max(float(xspec_arr[0]), min(start, stop))
    hi = min(float(xspec_arr[-1]), max(start, stop))

    if njuncs == 0:
        if lo >= hi:
            return np.float64(0.0) if pspec_was_1d else np.zeros(nspecs, dtype=np.float64)
        u = np.unique(np.concatenate(([lo, hi], xspec_arr[(xspec_arr > lo) & (xspec_arr < hi)])))
        S_u = interp1d(xspec_arr, Pspec_2d, kind="linear", axis=0, bounds_error=False, fill_value=0.0)(u)
        p_int = trapezoid(S_u, x=u, axis=0)
        return p_int[0] if pspec_was_1d else p_int

    if lo >= hi:
        return np.zeros((nspecs, njuncs), dtype=np.float64)

    u = np.unique(np.concatenate(([lo, hi], xspec_arr[(xspec_arr > lo) & (xspec_arr < hi)], xEQE_arr[(xEQE_arr > lo) & (xEQE_arr < hi)])))
    S_u = interp1d(xspec_arr, Pspec_2d, kind="linear", axis=0, bounds_error=False, fill_value=0.0)(u)  # (n_u, nspecs)
    E_u = interp1d(xEQE_arr, EQE_arr, kind="linear", axis=0, bounds_error=False, fill_value=0.0)(u)  # (n_u, njuncs)
    tw = np.zeros_like(u)
    du = np.diff(u)
    tw[:-1] += 0.5 * du
    tw[1:] += 0.5 * du
    weights = E_u * (tw * u * JCONST)[:, np.newaxis]  # (n_u, njuncs)
    return S_u.T @ weights  # (nspecs, njuncs)


@lru_cache(maxsize=100)
def JdbFromEg(TC: float, Eg: float, dbsides: float = 1.0, method: str | None = None) -> float:
    """
    return the detailed balance dark current
    assuming a square EQE
    Eg[=]eV
    TK[=]K
    returns Jdb[=]A/cm^2

    optional parameters
    method: 'gamma'
    dbsides:    single-sided->1.  bifacial->2.
    see special functions, Andrews p.71
    see Geisz EL&LC paper, King EUPVSEC
    """
    EgkT = Eg / convert.Vth(TC)
    TKlocal = convert.TK(TC)

    if str(method).lower() == "gamma":
        # use special function incomplete gamma
        # gamma(3)=2.0 not same incomplete gamma as in Igor
        Jdb = DB_PREFIX * TKlocal**3.0 * gammaincc(3.0, EgkT) * 2.0 * dbsides
    else:
        # Jdb as in Geisz et al.
        Jdb = DB_PREFIX * TKlocal**3.0 * (EgkT * EgkT + 2.0 * EgkT + 2.0) * np.exp(-EgkT) * dbsides  # units from DB_PREFIX

    return Jdb


@lru_cache(maxsize=100)
def EgFromJdb(TC: float, Jdb: float, Eg: float = 1.0, eps: float = 1e-6, itermax: int = 100, dbsides: float = 1.0) -> float:
    """Invert step-absorptance detailed-balance saturation current Jdb [A/cm^2] to bandgap Eg [eV]."""
    if not math.isfinite(Jdb) or Jdb <= 0.0:
        raise ValueError(f"EgFromJdb requires positive finite Jdb [A/cm^2], got {Jdb}")

    Vthlocal = convert.Vth(TC)
    TKlocal = convert.TK(TC)
    x0 = max(Eg / Vthlocal, 0.0)
    off = np.log(2.0 * dbsides * DB_PREFIX * TKlocal**3.0 / Jdb)
    count = 0

    while count < itermax:
        if x0 <= -1.0:
            x1 = off
        else:
            x1 = off + np.log(1.0 + x0 + x0 * x0 / 2.0)
        if not math.isfinite(x1):
            raise ValueError(f"EgFromJdb encountered non-finite iterate for Jdb={Jdb}, TC={TC}")
        tol = abs((x1 - x0) / x0) if x0 != 0.0 else abs(x1 - x0)
        if tol < eps:
            return float(x1 * Vthlocal)
        x0 = x1
        count += 1
    raise ValueError(f"EgFromJdb did not converge within {itermax} iterations for Jdb={Jdb}, TC={TC}")


def ensure_numpy_2drow(array: np.ndarray | pd.Index | pd.DataFrame | Sequence[int | float]) -> np.ndarray:

    # ensure numpy
    array = np.array(array)
    # ensure 2D
    array = array.reshape(1, -1) if array.ndim == 1 else array
    return array


def ensure_numpy_2dcol(array: np.ndarray | pd.Index | pd.DataFrame | Sequence[int | float]) -> np.ndarray:

    # ensure numpy
    array = np.array(array)
    # ensure 2D
    array = array.reshape(-1, 1) if array.ndim == 1 else array
    return array


class EQE:
    """
    EQE object
    It creates a class containing nth junctions EQEs and Luminescent Coupling
    between junctions. The contribution can be studied interectively using
    ipywidgets under a notebook.
    NOTE: Wavelengths become an N\times1 array, and EQE becomes an N\timesM array
    (N is the number of wavelength points, and M is the number of junctions).
    Each row corresponds to a single wavelength entry; columns represent
    different junctions.

    The methods
    self.control create the self.ui to adjust the LC
    self.plot function to plot
    sle.Jdb: it gets the reverse saturation current.

    """

    def __init__(self, wavelength: np.ndarray | pd.Index, eqe: np.ndarray | pd.DataFrame, name: str = "EQE", sjuncs: list[str] | None = None):
        """It creats the EQE class. ntegrate over spectrum or spectra
        rawEQE (numpy.array):  2D(lambda)(junction) raw input rawEQE (not LC corrected)
        xEQE(numpy.array)      xEQE        # wavelengths [nm] for rawEQE data
        name (str):            name of EQE object, sample
        sjuncs [list of str]:          labels for the junctions, if None it is self generated

        The number of junctions is created from the dimension of the EQE
        self.njuncs = njuncs    # number of junctions
        """
        wl_arr = np.asarray(wavelength, dtype=np.float64).ravel()
        eqe_arr = ensure_numpy_2dcol(np.asarray(eqe, dtype=np.float64))
        if wl_arr.shape[0] != eqe_arr.shape[0]:
            raise ValueError(f"shape of wavelength and eqe doesn't match: {wl_arr.shape[0]} != {eqe_arr.shape[0]}")

        if wl_arr.size > 1 and not np.all(np.diff(wl_arr) > 0):
            order = np.argsort(wl_arr, kind="stable")
            wl_arr = wl_arr[order]
            eqe_arr = eqe_arr[order, :]
            if np.any(np.diff(wl_arr) == 0):
                raise ValueError("EQE wavelength array contains duplicate entries")

        self.wavelength = ensure_numpy_2dcol(wl_arr)
        self.eqe = eqe_arr

        self.name = name

        self.njuncs = self.eqe.shape[1]  # number of junction

        if sjuncs is None:
            self.sjuncs = [ordinal(junc + 1) for junc in range(self.njuncs)]
        else:
            self.sjuncs = sjuncs  # names of junctions

        self.corrEQE = np.empty_like(self.eqe)  # luminescent coupling corrected EQE same size as rawEQE
        self.etas = np.zeros((self.njuncs, 3), dtype=np.float64)  # LC factor for next junctions
        self.LCcorr()  # calculate LC with zero etas
        self.spectra = None
        self._spec_wl: np.ndarray | None = None
        self._spec_full: np.ndarray | None = None

    def add_spectra(self, wavelength: np.ndarray | None = None, spectra: np.ndarray | None = None) -> None:
        """
        Add spectral irradiance data [W/m^2/nm]. Defaults to ASTM G173 AM1.5G ('global').

        Keeps the native spectral wavelength grid for accurate union-grid
        integration in 'Jint()', while also populating 'self.spectra' interpolated
        onto 'self.wavelength' (zero-padded outside the spectral range).
        """
        if wavelength is None and spectra is None:
            wavelength = wvl
            spectra = AM15G

        if wavelength is None or spectra is None:
            raise ValueError("Provide wavelength and spectra")

        wl_s = np.asarray(wavelength, dtype=np.float64).ravel()
        spec_s = ensure_numpy_2dcol(np.asarray(spectra, dtype=np.float64))
        if wl_s.shape[0] != spec_s.shape[0]:
            raise ValueError("Wavelength and Spectra need to have the same number of rows")

        if wl_s.size > 1 and not np.all(np.diff(wl_s) > 0):
            order = np.argsort(wl_s, kind="stable")
            wl_s = wl_s[order]
            spec_s = spec_s[order, :]
            if np.any(np.diff(wl_s) == 0):
                raise ValueError("Spectral wavelength array contains duplicate entries")

        self._spec_wl = wl_s
        self._spec_full = spec_s

        # Populate self.spectra on self.wavelength for backwards compatibility
        f_spec_interp = interp1d(wl_s, spec_s, kind="linear", axis=0, bounds_error=False, fill_value=0.0)
        self.spectra = f_spec_interp(self.wavelength.flatten())

    def _native_spectra(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(wavelength [nm], spectra) at the native resolution of 'add_spectra', or (None, None).

        The native copy is valid only while 'self.spectra' still holds its
        interpolation onto 'self.wavelength'. Once 'spectra' has been reassigned
        or edited in place, the copy is dropped and 'spectra' is used as given.
        """
        sw = getattr(self, "_spec_wl", None)
        S_full = getattr(self, "_spec_full", None)
        if sw is None or S_full is None:
            return None, None
        if self.spectra is not None:
            current = np.asarray(self.spectra)
            x = self.wavelength.flatten()
            if current.shape == (x.size, S_full.shape[1]):
                expected = interp1d(sw, S_full, kind="linear", axis=0, bounds_error=False, fill_value=0.0)(x)
                if np.array_equal(current, expected, equal_nan=True):
                    return sw, S_full
        self._spec_wl = None
        self._spec_full = None
        return None, None

    def add_eqe(self, wavelength_add: np.ndarray, eqe_add: np.ndarray, sjuncs: str | None = None) -> None:
        """
        Add eqe. Merge wavelength, inpolate eqe and fill extrapolation with 0. Assumes that the all EQE are part of a multijunction device.

        Args:
            eqe (optional): Wavelength of the new EQE
            eqe (optional): New EQE data
        """
        # ensure numpy and ascending wavelength order for added EQE
        wl_add_1d = np.asarray(wavelength_add, dtype=np.float64).ravel()
        eqe_add_2d = ensure_numpy_2dcol(np.asarray(eqe_add, dtype=np.float64))
        if wl_add_1d.shape[0] != eqe_add_2d.shape[0]:
            raise ValueError(f"shape of wavelength_add and eqe_add doesn't match: {wl_add_1d.shape[0]} != {eqe_add_2d.shape[0]}")
        if wl_add_1d.size > 1 and not np.all(np.diff(wl_add_1d) > 0):
            order = np.argsort(wl_add_1d, kind="stable")
            wl_add_1d = wl_add_1d[order]
            eqe_add_2d = eqe_add_2d[order, :]
        wavelength_add = ensure_numpy_2dcol(wl_add_1d)
        eqe_add = eqe_add_2d

        old_wavelength = self.wavelength.flatten()
        sw_native, S_native = self._native_spectra()

        # Combine old and new wavelength
        combined_wavelength = np.sort(np.unique(np.concatenate((self.wavelength, wavelength_add))))

        # Create interpolation functions with fill_value=0 for extrapolation
        f_interp_eqe = interp1d(self.wavelength.flatten(), self.eqe, kind="linear", fill_value=0, bounds_error=False, axis=0)
        f_interp_eqe_add = interp1d(wavelength_add.flatten(), eqe_add, kind="linear", fill_value=0, bounds_error=False, axis=0)

        # Interpolate and fill missing values
        eqe_interp = f_interp_eqe(combined_wavelength)
        eqe_add_interp = f_interp_eqe_add(combined_wavelength)

        old_etas = self.etas.copy()

        # Assign new values
        self.wavelength = ensure_numpy_2dcol(combined_wavelength)
        self.eqe = np.hstack([eqe_interp, eqe_add_interp])

        self.njuncs = self.eqe.shape[1]  # number of junction

        for k in range(eqe_add.shape[1]):
            if sjuncs is None:
                self.sjuncs.append(ordinal(self.njuncs - eqe_add.shape[1] + k + 1))
            else:
                self.sjuncs.append(sjuncs)

        self.corrEQE = np.empty_like(self.eqe)  # luminescent coupling corrected EQE same size as rawEQE
        self.etas = np.zeros((self.njuncs, 3), dtype=np.float64)  # preserve existing LC factors
        self.etas[: old_etas.shape[0], :] = old_etas
        self.LCcorr()  # recalculate LC-corrected EQE

        if sw_native is not None:
            self.spectra = interp1d(sw_native, S_native, kind="linear", axis=0, bounds_error=False, fill_value=0.0)(combined_wavelength)
        elif self.spectra is not None and np.shape(self.spectra)[0] == old_wavelength.size:
            # spectra assigned or edited directly: carry them over to the merged grid
            self.spectra = interp1d(old_wavelength, np.asarray(self.spectra, dtype=np.float64), kind="linear", axis=0, bounds_error=False, fill_value=0.0)(combined_wavelength)

    def calc_Eg_Rau(
        self,
        return_sigma: bool = True,
        fit_gaussian: bool = True,
        plot_fits: bool = False,
        threshold: float = float(np.exp(-2)),
    ) -> tuple[list[float], list[float | None]]:
        r"""Estimate photovoltaic bandgap Eg [eV] and edge width sigma [eV] per EQE column.

        Follows U. Rau et al., Phys. Rev. Applied 7, 044016 (2017): locates the
        primary absorption edge from 'P(E) = dEQE/dE' in photon-energy space
        within the window 'P(E) >= threshold * max(P)'.

        - When 'fit_gaussian=True' (default), fits the integrated Gaussian edge
          model 'A * Phi((E - Eg) / sigma)' directly to the rising EQE edge
          around the 'P(E)' peak. This returns the true Gaussian width 'sigma'
          (matching 'Junction.Jdb(..., theta=2.0)') without truncated-window or
          coarse-grid finite-difference bias.
        - When 'fit_gaussian=False', returns the non-parametric Rau first moment
          'int E * P(E) dE / int P(E) dE' over the window and 'sigma = None'.
        - Discontinuous ideal step EQEs return the step midpoint energy and
          'sigma = 0.0' (when 'fit_gaussian=True' and 'return_sigma=True').
        """
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")

        E_raw = np.asarray(convert.wavelength_to_photonenergy(self.wavelength.flatten()), dtype=np.float64)
        order = np.argsort(E_raw)
        E_all = E_raw[order]
        eqe_sorted = np.asarray(self.eqe, dtype=np.float64)[order, :]

        bandgaps: list[float] = []
        sigmas: list[float | None] = []

        for i in range(eqe_sorted.shape[1]):
            y_all = eqe_sorted[:, i]
            if E_all.size < 2 or not np.any(np.isfinite(y_all)) or np.nanmax(y_all) <= 0.0:
                raise ValueError(f"EQE column {i} has insufficient or non-positive data for calc_Eg_Rau")

            # Differentiate with respect to photon energy E (not array index)
            P_all = np.gradient(y_all, E_all)
            pos_mask = np.isfinite(P_all) & (P_all > 0.0)
            if not np.any(pos_mask):
                raise ValueError(f"EQE column {i} has no rising absorption edge (dEQE/dE <= 0 everywhere)")

            k_peak = int(np.nanargmax(np.where(pos_mask, P_all, -np.inf)))
            P_max = float(P_all[k_peak])
            p_cut = threshold * P_max

            # Contiguous window around k_peak where P(E) >= threshold * P_max
            lo = k_peak
            while lo > 0 and P_all[lo - 1] >= p_cut:
                lo -= 1
            hi = k_peak
            while hi < len(P_all) - 1 and P_all[hi + 1] >= p_cut:
                hi += 1

            Ew = E_all[lo : hi + 1]
            Pw = P_all[lo : hi + 1]

            # Detect discontinuous step EQE (>= 85% of total EQE jump occurs within +-1 grid step and narrow window)
            y_span = float(np.nanmax(y_all) - np.nanmin(y_all))
            local_jump = float(y_all[min(k_peak + 1, len(E_all) - 1)] - y_all[max(k_peak - 1, 0)])
            is_ideal_step = Ew.size <= 2 and y_span > 0.0 and local_jump >= 0.99 * y_span

            if is_ideal_step:
                l_step = max(k_peak - 1, 0)
                h_step = min(k_peak + 1, len(E_all) - 1)
                E_step = E_all[l_step : h_step + 1]
                P_step = np.maximum(P_all[l_step : h_step + 1], 0.0)
                denom_s = float(trapezoid(P_step, E_step)) if E_step.size >= 2 else 0.0
                bg = float(trapezoid(E_step * P_step, E_step) / denom_s) if denom_s > 0.0 else float(E_all[k_peak])
                sig: float | None = (0.0 if fit_gaussian else None) if return_sigma else None
                bandgaps.append(bg)
                sigmas.append(sig)
                continue

            denom_w = float(trapezoid(Pw, Ew)) if Ew.size >= 2 else 0.0
            mean_w = float(trapezoid(Ew * Pw, Ew) / denom_w) if denom_w > 0.0 else float(np.average(Ew, weights=Pw))

            if not fit_gaussian:
                bandgaps.append(mean_w)
                sigmas.append(None)
                continue

            std_w = float(np.sqrt(np.sum(Pw * (Ew - mean_w) ** 2) / np.sum(Pw))) if Ew.size >= 2 else 0.0
            local_dE = float(E_all[min(k_peak + 1, len(E_all) - 1)] - E_all[max(k_peak - 1, 0)]) * 0.5
            sigma0 = max(std_w / 0.88, 0.5 * local_dE, 1e-3)

            # Build monotonic rising-edge segment around k_peak (stopping before any upper-junction falling edge)
            s_lo = lo
            while s_lo > 0 and E_all[s_lo - 1] >= mean_w - 4.0 * sigma0 and P_all[s_lo - 1] >= -0.02 * P_max:
                s_lo -= 1
            s_hi = hi
            while s_hi < len(E_all) - 1 and E_all[s_hi + 1] <= mean_w + 4.0 * sigma0 and P_all[s_hi + 1] >= -0.02 * P_max:
                s_hi += 1

            Es = E_all[s_lo : s_hi + 1]
            Ys = y_all[s_lo : s_hi + 1]

            if Es.size < 3:
                bandgaps.append(mean_w)
                sigmas.append(0.0 if return_sigma else None)
                continue

            A0 = max(float(np.max(Ys)), 1e-6)
            lower = [0.05 * A0, float(E_all[0]) * 0.5, 1e-5]
            upper = [2.0 * max(A0, 1.0), float(E_all[-1]) * 1.5, 0.5]
            # a broad edge can put sigma0 (or mean_w) beyond the fit bounds, where
            # least_squares refuses to start
            x0 = np.clip([A0, mean_w, sigma0], lower, upper)
            fit_lsq = least_squares(
                lambda p: p[0] * ndtr((Es - p[1]) / p[2]) - Ys,
                x0=x0,
                bounds=(lower, upper),
            )
            A_fit, bg, sig_val = float(fit_lsq.x[0]), float(fit_lsq.x[1]), float(abs(fit_lsq.x[2]))

            if plot_fits:
                P_pos = P_all[pos_mask]
                E_pos = E_all[pos_mask]
                P_norm = convert.normalize(P_pos)
                x_fit = np.linspace(float(Es[0]), float(Es[-1]), 100)
                gauss_amp = A_fit / (np.sqrt(2.0 * np.pi) * sig_val) / P_max
                y_fit = _gaussian(x_fit, gauss_amp, bg, sig_val)
                _, ax = plt.subplots(1, 2, layout="constrained")
                ax[0].plot(E_all, y_all)
                ax[0].plot(x_fit, A_fit * ndtr((x_fit - bg) / sig_val), "--")
                ax[1].plot(E_pos, P_norm, "--")
                ax[1].plot(x_fit, y_fit)
                ax[1].plot(Ew, Pw / P_max, ".r")
                ax[1].plot(E_all[k_peak], 1.0, "r*")
                ax[1].plot(Ew[0], Pw[0] / P_max, "g*")
                ax[1].plot(Ew[-1], Pw[-1] / P_max, "b*")
                ax[0].set_xlabel(r"Photon energy $E_{\nu}$ [eV]")
                ax[0].set_ylabel(r"EQE")
                ax[1].set_xlabel(r"Photon energy $E_{\nu}$ [eV]")
                ax[1].set_ylabel(r"$\frac{{\mathrm{d}}EQE}{{\mathrm{d}}E_{\nu}}$")

            bandgaps.append(bg)
            sigmas.append(sig_val if return_sigma else None)

        return bandgaps, sigmas

    def LCcorr(self, junc: int | None = None, dist: int | None = None, val: float | None = None) -> None:
        """Apply luminescent-coupling correction to 'self.eqe' -> 'self.corrEQE'.

        Follows Steiner et al., IEEE J. Photovolt. 3, 879 (2013) and Geisz et al.,
        IEEE J. Photovolt. 5, 1827 (2015), Sec. III. Note that 'self.etas'
        parametrizes the exact linear inversion in terms of the *raw* upper-junction
        EQEs as fitted by nulling the corrected subcell EQE in the absorption bands
        of optically thick upper junctions (for a pure small-signal cascade
        g_{12}, g_{23} without direct 1->3 coupling, the effective entries are
        'etas[2, 0] = g_{23} * (1 + g_{12})' and 'etas[2, 1] = 0').
        """
        # change one eta[junc,dist] value
        # calculate LC corrected EQE
        etas = self.etas
        # with self.debugout: print(junc,dist,val)
        if junc is None or dist is None or val is None:
            pass
        else:
            etas[junc, dist] = val  # assign value
            # with self.debugout: print('success')
        raw = self.eqe
        ## TODO: This should be a nested loop, with a break for 10th junctions?
        if self.njuncs == 1:
            self.corrEQE = raw
        else:
            for ijunc in range(self.njuncs):
                if ijunc == 0:  # 1st ijunction
                    self.corrEQE[:, ijunc] = raw[:, ijunc]
                elif ijunc == 1:  # 2nd ijunction
                    denom = 1.0 + etas[ijunc, 0]
                    self.corrEQE[:, ijunc] = raw[:, ijunc] * denom - raw[:, ijunc - 1] * etas[ijunc, 0]
                elif ijunc == 2:  # 3rd ijunction
                    denom = 1.0 + etas[ijunc, 0] * (1.0 + etas[ijunc, 1])
                    self.corrEQE[:, ijunc] = raw[:, ijunc] * denom - raw[:, ijunc - 1] * etas[ijunc, 0] - raw[:, ijunc - 2] * etas[ijunc, 0] * etas[ijunc, 1]
                else:  # higher ijunctions
                    denom = 1.0 + etas[ijunc, 0] * (1.0 + etas[ijunc, 1] * (1.0 + etas[ijunc, 2]))
                    self.corrEQE[:, ijunc] = (
                        raw[:, ijunc] * denom
                        - raw[:, ijunc - 1] * etas[ijunc, 0]
                        - raw[:, ijunc - 2] * etas[ijunc, 0] * etas[ijunc, 1]
                        - raw[:, ijunc - 3] * etas[ijunc, 0] * etas[ijunc, 1] * etas[ijunc, 2]
                    )

    def Jdb(self, TC: float, Eguess: float | Sequence[float] | None = 1.0, kTfilter: int = 3, dbug: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Calculate detailed-balance Jdb [A/cm^2] and step-equivalent Eg [eV] from 'self.corrEQE'.

        Applies the iterative sub-gap noise filter at 'Eg - kTfilter * Vth' per
        column (note: for broad absorption tails with sigma >~ 1.5*Vth this also
        truncates sub-gap tail emission). Stores 'self.Egs = Egnew'.
        """
        Jdb_out, Egnew, _ = _jdb_from_eqe_columns(self.corrEQE, self.wavelength.flatten(), TC, Eguess=Eguess, kTfilter=kTfilter, debug=dbug)
        self.Egs = Egnew
        return Jdb_out, Egnew

    def Jint(
        self,
        enforce_all_combinations: bool | None = None,
        *,
        pairwise: bool = False,
    ) -> np.ndarray:
        """
        Integrate corrected EQE columns against the loaded spectra.

        For corrected EQE column ``Q_i(lambda)`` and spectral-irradiance
        column ``S_j(lambda)``, the current density is

            J_ij = 0.1 * int Q_i(lambda) * S_j(lambda) / E_ph(lambda) d lambda

        Here ``S_j`` is in W/m^2/nm, ``E_ph`` is the photon energy in eV, and
        ``J_ij`` is in mA/cm^2. The factor 0.1 converts A/m^2 to mA/cm^2 after
        the electron charge cancels in the photon-flux-to-current conversion.

        Default mode (``pairwise=False``): default, every EQE column against every spectrum
        ----------------------------------------------------
        ``Jint()`` and ``Jint(pairwise=False)`` integrate every EQE column
        with every spectrum column.
        The result is always a 2-D array with shape
        ``(n_eqe, n_spectra)``:

            EQE columns   Spectra   Output shape   Meaning
            1             1         (1, 1)         one junction, one spectrum
            1             m         (1, m)         one junction, every spectrum
            n             1         (n, 1)         every junction, one spectrum
            n             m         (n, m)         every junction, every spectrum

        Singleton dimensions are retained. For two junctions and two spectra,
        for example:

            Jint() = [[J_top_s0, J_top_s1],
                      [J_bot_s0, J_bot_s1]]     # shape (2, 2)

        Pairwise mode (``pairwise=True``): only matching EQE/spectrum columns
        --------------------------------------------------------
        ``Jint(pairwise=True)`` requires the same number of EQE and spectrum
        columns. It integrates EQE column ``i`` only with spectrum column
        ``i`` and returns a 1-D array:

            Jint(pairwise=True) = [J_00, J_11, J_22]  # 3 matched pairs

        This is equivalent to taking the diagonal of the default result; all
        off-diagonal combinations are omitted. For the two-junction example
        above, pairwise mode returns ``[J_top_s0, J_bot_s1]``. It does not
        evaluate both junctions under both spectra and it does not sum their
        currents.

        Use pairwise mode only when EQE column ``i`` and spectrum column ``i``
        intentionally represent the same observation, such as one
        temperature-adjusted EQE paired with each time-step spectrum. For an
        ordinary multijunction EQE evaluated under one or more spectra, use
        the default mode.

        Args:
            enforce_all_combinations: Deprecated compatibility argument. ``True``
                selects every combination. ``False`` selects the historical
                behavior: pair matching columns when their counts are equal,
                otherwise use every combination. New code should use ``pairwise``.
                Its default ``None`` only means that this deprecated argument
                was not supplied; it is not a third calculation mode.
            pairwise: ``False`` returns every combination as a 2-D matrix.
                ``True`` returns only matching column pairs as a 1-D array.
                Pairwise mode raises ``ValueError`` when the EQE and spectrum
                column counts differ. ``None`` is not accepted.

        Note:
            For an :class:`EQET`, rows of the default matrix correspond to
            measured EQE temperatures rather than junctions. Use
            :meth:`EQET.get_current_for_temperature` to interpolate current at
            each time step's cell temperature.

        Returns:
            The ``(n_eqe, n_spectra)`` current-density matrix, or the ``(n,)``
            vector of matching pairs when ``pairwise=True``.
        """

        if self.spectra is None:
            raise ValueError("Load spectral information first.")

        if not isinstance(pairwise, bool):
            raise TypeError("Jint() pairwise must be True or False")

        if enforce_all_combinations is not None and pairwise:
            raise TypeError("Jint() cannot combine the deprecated positional argument with pairwise=True")

        if enforce_all_combinations is not None:
            warnings.warn(
                "Jint(enforce_all_combinations=...) and positional booleans are deprecated; use pairwise=False (matrix, default) or pairwise=True.",
                DeprecationWarning,
                stacklevel=2,
            )
            pairwise = not enforce_all_combinations and self.eqe.shape[1] == self.spectra.shape[1]

        n_spec = self.spectra.shape[1]
        if pairwise and self.eqe.shape[1] != n_spec:
            raise ValueError(
                f"Jint(pairwise=True) needs one spectrum per EQE column, got {self.eqe.shape[1]} EQE columns and {n_spec} spectra. Use pairwise=False for the full (n_eqe, n_spectra) matrix."
            )

        x = self.wavelength.flatten()
        sw, S_full = self._native_spectra()
        if sw is None or S_full is None:
            sw = x
            S_full = np.asarray(self.spectra, dtype=np.float64)

        lo = max(float(x[0]), float(sw[0]))
        hi = min(float(x[-1]), float(sw[-1]))
        if lo >= hi:
            return np.zeros(n_spec, dtype=np.float64) if pairwise else np.zeros((self.eqe.shape[1], n_spec), dtype=np.float64)

        # Integrate on the union of spectral and EQE wavelength nodes inside [lo, hi]
        u = np.unique(np.concatenate(([lo, hi], sw[(sw > lo) & (sw < hi)], x[(x > lo) & (x < hi)])))
        S_u = interp1d(sw, S_full, kind="linear", axis=0, bounds_error=False, fill_value=0.0)(u)  # (n_u, n_spec)
        E_u = interp1d(x, self.corrEQE, kind="linear", axis=0, bounds_error=False, fill_value=0.0)(u)  # (n_u, n_eqe)
        tw = np.zeros_like(u)
        du = np.diff(u)
        tw[:-1] += 0.5 * du
        tw[1:] += 0.5 * du
        weights = E_u * (tw * u * JCONST)[:, np.newaxis]  # (n_u, n_eqe)

        if pairwise:
            return np.einsum("ui,ui->i", weights, S_u)  # shape (n,)

        return weights.T @ S_u  # shape (n_eqe, n_spectra)

    def plot(
        self,
        Pspec: str | np.ndarray = "global",
        ispec: int = 0,
        specname: str | None = None,
        xspec: np.ndarray = wvl,
        size: str = "x-large",
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
    ) -> tuple[plt.Axes, plt.Axes]:
        # plot EQE on top of a spectrum
        rnd2 = 100

        if fig is None or ax is None:
            fig, ax = plt.subplots()
        if len(fig.get_axes()) > 1:
            fig.set_layout_engine("constrained")

        for i in range(self.njuncs):
            rlns = ax.plot(self.wavelength, self.eqe[:, i], lw=1, ls="--", marker="", label="_" + self.sjuncs[i])
            ax.plot(self.wavelength, self.corrEQE[:, i], lw=3, c=rlns[0].get_color(), marker="", label=self.sjuncs[i])
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(math.floor(self.wavelength[0].min() / rnd2) * rnd2, math.ceil(self.wavelength[-1].max() / rnd2) * rnd2)
        ax.set_ylabel("EQE", size=size)  # Add a y-label to the axes.
        ax.set_xlabel("Wavelength (nm)", size=size)  # Add an x-label to the axes.
        ax.set_title(self.name + " EQE", size=size)
        ax.axhline(0, lw=0.5, ls="--", c="black", label="_hzero")
        ax.axhline(1, lw=0.5, ls="--", c="black", label="_hone")

        rax = ax.twinx()  # right axis
        # check spectra input
        if type(Pspec) is str:  # optional string space, global, direct
            if Pspec in dfrefspec.columns:
                specname = Pspec
                Pspec = dfrefspec[Pspec].to_numpy(dtype=np.float64, copy=True)
            else:
                specname = dfrefspec.columns[ispec]
                Pspec = dfrefspec.to_numpy(dtype=np.float64, copy=True)  # just use refspec instead of error

        if np.any(Pspec):
            Pspec = np.array(Pspec, dtype=np.float64)
            if not specname:
                specname = "spectrum" + str(ispec)
            if Pspec.ndim == 2:
                Pspec = Pspec[:, ispec]  # slice 2D numpy to 1D
            rax.fill_between(xspec, Pspec, step="mid", alpha=0.2, color="grey", label="fill")
            rax.plot(xspec, Pspec, c="grey", lw=0.5, marker="", label=specname)
            rax.set_ylabel("Irradiance (W/m2/nm)", size=size)  # Add a y-label to the axes.
            rax.set_ylim(0, 2)
            # rax.legend(loc=7)
        return ax, rax

    def plot_sr(self) -> tuple[plt.Figure, plt.Axes]:
        # plot EQE on top of a spectrum

        sr = self.eqe * 1 / np.asarray(convert.wavelength_to_photonenergy(self.wavelength))
        fig, ax = plt.subplots()
        ax.plot(self.wavelength, sr)
        return fig, ax


class EQET(EQE):

    def __init__(self, wavelength: np.ndarray | pd.Index, eqe: np.ndarray | pd.DataFrame, temperature: np.ndarray | float | Sequence[float], name: str = "EQE", sjuncs: list[str] | None = None):

        # ensure 1-D numpy array (also handles 0-d numpy scalars)
        temperature = np.atleast_1d(np.asarray(temperature)).ravel()

        super().__init__(wavelength, eqe, name, sjuncs)

        self.temperature = temperature

        self.method_dict = {
            "interpolate": self._interpolate_qe,
            "model": self._qe_from_model,
        }

        self._sort_by_temperature()

        # init empty paramters
        self.spectra = None
        self.topcell_bandgap_temperature_model = None
        self.topcell_sigma_temperature_model = None
        self.bottomcell_bandgap_temperature_model = None
        self.bottomcell_sigma_temperature_model = None

    def _sort_by_temperature(self) -> None:
        temperature_sorter = np.argsort(self.temperature)
        self.temperature = self.temperature[temperature_sorter]
        self.eqe = self.eqe[:, temperature_sorter]
        self.corrEQE = self.corrEQE[:, temperature_sorter]

    def _qe_from_model(self, temperature: np.ndarray) -> EQET:
        raise NotImplementedError("EQET method 'model' is not implemented yet; use method='interpolate'")

    # def controls(self, Pspec='global', ispec=0, specname=None, xspec=wvl):
    #     '''
    #     use interactive_output for GUI in IPython
    #     '''
    #     tand_layout = widgets.Layout(width= '300px', height='40px')
    #     vout_layout = widgets.Layout(width= '180px', height='40px')
    #     junc_layout = widgets.Layout(display='flex',
    #                 flex_flow='row',
    #                 justify_content='space-around')
    #     multi_layout = widgets.Layout(display='flex',
    #                 flex_flow='row',
    #                 justify_content='space-around')

    #     replot_types = [widgets.widgets.widget_float.BoundedFloatText,
    #                     widgets.widgets.widget_int.BoundedIntText,
    #                     widgets.widgets.widget_int.IntSlider,
    #                     widgets.widgets.widget_float.FloatSlider,
    #                     widgets.widgets.widget_float.FloatLogSlider]

    #     def on_EQEchange(change):
    #         # function for changing values
    #         old = change['old'] #old value
    #         new = change['new'] #new value
    #         owner = change['owner'] #control
    #         value = owner.value
    #         desc = owner.description
    #         #with self.debugout: print('Mcontrol: ' + desc + '->', value)
    #         #self.set(**{desc:value})

    #     def on_EQEreplot(change):
    #         # change info
    #         fast=True
    #         if type(change) is widgets.widgets.widget_button.Button:
    #             owner = change
    #         else: # other controls
    #             owner = change['owner'] #control
    #             value = owner.value
    #         desc = owner.description
    #         if desc == 'Recalc': fast = False

    #         #recalculate
    #         ts = time()
    #         if desc[:3] == 'eta':
    #             junc, dist = parse('eta{:1d}{:1d}',desc)
    #             self.LCcorr(junc, dist, value) #replace one value and recalculate LC
    #             specname = None
    #         elif desc == 'spec':
    #             if value in dfrefspec.columns:
    #                 specname = value
    #                 Pspec = dfrefspec[specname].to_numpy(dtype=np.float64, copy=True)
    #         else:
    #             VoutBox.clear_output()
    #             with VoutBox: print(desc)
    #             return 0

    #         with Rout: # right output device -> light
    #             #replot
    #             lines = ax.get_lines()
    #             for line in lines:
    #                 linelabel=line.get_label()
    #                 #with self.debugout: print(linelabel)
    #                 for i in range(self.njuncs):
    #                     if linelabel == self.sjuncs[i]:
    #                          line.set_data(self.wavelength, self.corrEQE[:,i]) #replot

    #             rlines = rax.get_lines()
    #             for line in rlines:
    #                 linelabel=line.get_label()
    #                 #with self.debugout: print(linelabel)
    #                 if linelabel in refnames:
    #                     if specname == None: #desc == 'spec'
    #                          specname = linelabel
    #                          Pspec = specname
    #                     else:
    #                         line.set_data(xspec, Pspec) #replot spectrum
    #                         for obj in rax.get_children():
    #                             if type(obj) is mpl.collections.PolyCollection: #contours
    #                                 if obj.get_label() == 'fill':
    #                                     obj.remove() #remove old fill
    #                         rax.fill_between(xspec, Pspec, step="mid", alpha=0.2, color='grey', label='fill')
    #                         line.set(label = specname) #relabel spectrum

    #         Jscs = self.Jint(Pspec, xspec)
    #         Jdbs, Egs = self.Jdb(25)
    #         OP = PintMD(Pspec, xspec)

    #         VoutBox.clear_output()
    #         with VoutBox:
    #             stext = (specname+' {0:6.2f} W/m2').format(OP)
    #             print('Eg = ',Egs, ' eV')
    #             print(stext)
    #             print('Jsc = ',Jscs[0], ' mA/cm^2')

    #         te = time()
    #         dt=(te-ts)
    #         with VoutBox:   print('Calc Time: {0:>6.2f} s'.format(dt))

    #     # summary line
    #     VoutBox = widgets.Output()
    #     VoutBox.layout.height = '70px'
    #     #with VoutBox: print('Summary')

    #     # Right output -> EQE plot
    #     Rout = widgets.Output()
    #     with Rout: # output device
    #         if plt.isinteractive:
    #             plt.ioff()
    #             restart = True
    #         else:
    #             restart = False
    #         ax, rax = self.plot(Pspec, ispec, specname, xspec)
    #         fig = ax.get_figure()
    #         fig.show()
    #         rlines = rax.get_lines()
    #         for line in rlines:
    #             linelabel=line.get_label()
    #             if linelabel in refnames:
    #                 specname = linelabel
    #         if restart: plt.ion()

    #     # tandem3T controls
    #     in_tit = widgets.Label(value='EQE: ', description='title')
    #     in_name = widgets.Text(value=self.name, description='name', layout=tand_layout,
    #         continuous_update=False)
    #     in_name.observe(on_EQEchange,names='value') #update values

    #     in_spec = widgets.Dropdown(value=specname, description='spec', layout=tand_layout,
    #         options=refnames)
    #     in_spec.observe(on_EQEreplot,names='value') #update values

    #     Hui = widgets.HBox([in_tit, in_name, in_spec])
    #     #in_Rs2T.observe(on_2Tchange,names='value') #update values

    #     in_eta = []
    #     elist0 = []
    #     elist1 = []
    #     elist2 = []
    #     # list of eta controls
    #     for i in range(self.njuncs) :
    #         if i > 0:
    #             in_eta.append(widgets.FloatSlider(value=self.etas[i,0], min=-0.2, max=1.5,step=0.001,
    #                 description='eta'+str(i)+"0",layout=junc_layout,readout_format='.4f'))
    #             j = len(in_eta)-1
    #             elist0.append(in_eta[j])
    #             in_eta[j].observe(on_EQEreplot,names='value')  #replot
    #         #if i > 1:
    #             in_eta.append(widgets.FloatSlider(value=self.etas[i,1], min=-0.2, max=1.5,step=0.001,
    #                 description='eta'+str(i)+"1",layout=junc_layout,readout_format='.4f'))
    #             j = len(in_eta)-1
    #             elist1.append(in_eta[j])
    #             in_eta[j].observe(on_EQEreplot,names='value')  #replot
    #             if i > 1:
    #                 in_eta[j].observe(on_EQEreplot,names='value')  #replot
    #             else:
    #                 in_eta[j].disabled = True
    #         #if i > 2:
    #             in_eta.append(widgets.FloatSlider(value=self.etas[i,2], min=-0.2, max=1.5,step=0.001,
    #                 description='eta'+str(i)+"2",layout=junc_layout,readout_format='.4f'))
    #             j = len(in_eta)-1
    #             elist2.append(in_eta[j])
    #             if i > 2:
    #                 in_eta[j].observe(on_EQEreplot,names='value')  #replot
    #             else:
    #                 in_eta[j].disabled = True
    #     etaui0 = widgets.HBox(elist0)
    #     etaui1 = widgets.HBox(elist1)
    #     etaui2 = widgets.HBox(elist2)

    #     #in_Rs2T.observe(on_2Treplot,names='value')  #replot
    #     #in_2Tbut.on_click(on_2Treplot)  #replot

    #     #EQE_ui = widgets.HBox(clist)
    #     #eta_ui = widgets.HBox(jui)

    #     ui = widgets.VBox([Rout, VoutBox, Hui, etaui0, etaui1, etaui2])
    #     self.ui = ui
    #     #in_2Tbut.click() #fill in MPP values

    #     # return entire user interface, dark and light graph axes for tweaking
    #     return ui, ax, rax

    def _interpolate_qe(self, temperature: np.ndarray) -> EQET:
        temp_arr = np.asarray(temperature, dtype=np.float64)
        t_min, t_max = float(self.temperature.min()), float(self.temperature.max())
        finite_t = temp_arr[np.isfinite(temp_arr)]
        if finite_t.size and (np.any(finite_t < t_min) or np.any(finite_t > t_max)):
            logger.warning(
                "EQET.get_eqe_at_temperature: requested temperature [{:.1f}, {:.1f}] C lies outside measured range [{:.1f}, {:.1f}] C; per-wavelength linear extrapolation does not shift the absorption edge.",
                float(finite_t.min()),
                float(finite_t.max()),
                t_min,
                t_max,
            )

        t_unique, inv = np.unique(self.temperature.astype(np.float64), return_inverse=True)
        if t_unique.size < self.temperature.size:
            eqe_u = np.column_stack([self.eqe[:, inv == k].mean(axis=1) for k in range(t_unique.size)])
        else:
            eqe_u = self.eqe

        if t_unique.size == 1:
            eqe_new = np.repeat(eqe_u, np.atleast_1d(temp_arr).size, axis=1)
            if np.ndim(temperature) == 0:
                eqe_new = eqe_new[:, 0]
        else:
            interp_func = interp1d(t_unique, eqe_u, axis=1, kind="linear", bounds_error=False, fill_value="extrapolate")
            eqe_new = interp_func(temperature)

        eqe_new = np.clip(eqe_new, 0, self.eqe.max())
        return EQET(self.wavelength, eqe_new, temperature)

    def get_eqe_at_temperature(self, temperature: np.ndarray | float, method: str = "interpolate") -> EQET:
        """Interpolate EQE columns to target temperature(s) [degC].

        Note: 'method="interpolate"' interpolates and extrapolates EQE linearly
        in temperature at each wavelength independently and clips to '[0, max(EQE)]';
        outside the measured temperature range the absorption edge is not
        energy-shifted.
        """
        if method not in self.method_dict:
            raise ValueError(f"Method '{method}' is not implemented. Available methods: {list(self.method_dict.keys())}")
        return self.method_dict[method](np.asarray(temperature))

    def get_current_for_temperature(
        self,
        target_temperature: float | Sequence[float] | np.ndarray,
        degrees: int | Sequence[int] | str | None = None,
    ) -> np.ndarray:
        r"""Calculate short-circuit current density J(T) [mA/cm^2] for each target temperature.

        Integrates each measured 'EQE(T_k, \lambda)' against its paired spectrum via
        'self.Jint()', then evaluates 'J(T)' at 'target_temperature[i]':

        - 'degrees=None' or '"interp"' (default): piecewise-linear interpolation
          in temperature between measured points, with linear extrapolation outside
          '[T_min, T_max]' from a local linear fit over the nearest measured
          temperatures.
        - 'degrees=d' (int): polynomial fit of degree 'min(d, n_unique - 1)' in
          centered/scaled temperature.
        - 'degrees=[...]' (sequence of ints): selects the lowest leave-one-out
          cross-validation MSE among candidate degrees capped at
          'max(1, n_unique - 2)' so exact-interpolant overfitting is excluded.
        """
        target_arr = np.atleast_1d(np.asarray(target_temperature, dtype=np.float64)).ravel()
        if self.spectra is None:
            raise ValueError("Load spectral information first.")

        if self.spectra.shape[1] != target_arr.shape[0]:
            raise ValueError("Spectral and temperature array not the same length.")

        # (n_measured_temps, n_spectra)
        current_density = self.Jint()
        n_spec = current_density.shape[1]

        # Collapse duplicate measured temperatures by averaging Jint rows
        t_unique, inv = np.unique(self.temperature.astype(np.float64), return_inverse=True)
        if t_unique.size < self.temperature.size:
            Y = np.vstack([current_density[inv == k, :].mean(axis=0) for k in range(t_unique.size)])
        else:
            Y = current_density  # (n_u, n_spec)
        n_u = t_unique.size

        finite_targets = target_arr[np.isfinite(target_arr)]
        if finite_targets.size and (np.any(finite_targets < t_unique[0]) or np.any(finite_targets > t_unique[-1])):
            logger.warning(
                "EQET.get_current_for_temperature: target temperatures [{:.1f}, {:.1f}] C extend outside measured range [{:.1f}, {:.1f}] C",
                float(finite_targets.min()),
                float(finite_targets.max()),
                float(t_unique[0]),
                float(t_unique[-1]),
            )

        if n_u == 1:
            return np.where(np.isfinite(target_arr), Y[0, :], np.nan)

        if isinstance(degrees, str) and degrees != "interp":
            raise ValueError(f"degrees must be None, 'interp', an int or a sequence of ints, got {degrees!r}")

        if degrees is None or isinstance(degrees, str):
            out = np.full(n_spec, np.nan, dtype=np.float64)
            fin = np.flatnonzero(np.isfinite(target_arr))
            if fin.size == 0:
                return out
            tf = target_arr[fin]
            Yf = Y[:, fin]

            # Piecewise-linear interpolation inside [t_min, t_max]
            idx_hi = np.clip(np.searchsorted(t_unique, tf, side="right"), 1, n_u - 1)
            idx_lo = idx_hi - 1
            t0 = t_unique[idx_lo]
            t1 = t_unique[idx_hi]
            frac = (tf - t0) / (t1 - t0)
            y_interp = Yf[idx_lo, np.arange(fin.size)] * (1.0 - frac) + Yf[idx_hi, np.arange(fin.size)] * frac

            # Local-linear extrapolation below t_unique[0] (span >= 25 C when dense)
            below = tf < t_unique[0]
            if np.any(below):
                k_lo = max(2, int(np.sum(t_unique <= t_unique[0] + 25.0)))
                k_lo = min(k_lo, n_u)
                tl = t_unique[:k_lo]
                tl_c = tl - tl.mean()
                denom = float(np.sum(tl_c**2))
                Yl = Yf[:k_lo, :][:, below]
                slope_lo = (tl_c @ Yl) / denom
                intercept_lo = Yl.mean(axis=0) - slope_lo * tl.mean()
                y_interp[below] = intercept_lo + slope_lo * tf[below]

            # Local-linear extrapolation above t_unique[-1]
            above = tf > t_unique[-1]
            if np.any(above):
                k_hi = max(2, int(np.sum(t_unique >= t_unique[-1] - 25.0)))
                k_hi = min(k_hi, n_u)
                th = t_unique[-k_hi:]
                th_c = th - th.mean()
                denom = float(np.sum(th_c**2))
                Yh = Yf[-k_hi:, :][:, above]
                slope_hi = (th_c @ Yh) / denom
                intercept_hi = Yh.mean(axis=0) - slope_hi * th.mean()
                y_interp[above] = intercept_hi + slope_hi * tf[above]

            out[fin] = y_interp
            return out

        # Polynomial fit in centered/scaled temperature u = (T - t_mean) / t_scale
        t_mean = float(t_unique.mean())
        t_scale = max(float(t_unique.std()), 1.0)
        u_meas = (t_unique - t_mean) / t_scale
        u_targ = (target_arr - t_mean) / t_scale

        if isinstance(degrees, (int, np.integer)):
            deg = int(degrees)
            if deg < 0:
                raise ValueError(f"degrees must be >= 0, got {deg}")
            max_identifiable = n_u - 1
            if deg > max(1, n_u - 2):
                logger.warning(
                    "EQET.get_current_for_temperature: degree={} with {} unique temperatures risks severe extrapolation error; consider degrees=None (piecewise-linear)",
                    deg,
                    n_u,
                )
            deg = min(deg, max_identifiable)
            V = np.vander(u_meas, N=deg + 1, increasing=True)
            coeffs, _, _, _ = np.linalg.lstsq(V, Y, rcond=None)  # (deg + 1, n_spec)
            u_pows = np.vander(u_targ, N=deg + 1, increasing=True)  # (n_spec, deg + 1)
            return np.einsum("ij,ji->i", u_pows, coeffs)

        deg_list = sorted({int(d) for d in degrees if int(d) >= 0})
        if not deg_list:
            raise ValueError("degrees list must contain at least one non-negative integer")

        # Cap polynomial degree at max(1, n_u - 2) (and <= n_u - 1 when n_u == 2)
        deg_cap = min(max(1, n_u - 2), n_u - 1)
        candidates = [d for d in deg_list if d <= deg_cap]
        if not candidates:
            candidates = [min(min(deg_list), deg_cap)]

        best_loo = np.full(n_spec, np.inf, dtype=np.float64)
        best_pred = np.full(n_spec, np.nan, dtype=np.float64)

        for deg in candidates:
            V = np.vander(u_meas, N=deg + 1, increasing=True)  # (n_u, deg + 1)
            Q, R = np.linalg.qr(V, mode="reduced")
            if np.linalg.matrix_rank(R) < deg + 1:
                continue
            h = np.sum(Q * Q, axis=1)  # leverage (n_u,)
            Y_hat = Q @ (Q.T @ Y)
            denom = np.maximum(1.0 - h[:, np.newaxis], 1e-12)
            if deg + 1 >= n_u:
                loo_mse = np.mean((Y - Y_hat) ** 2, axis=0)
            else:
                loo_mse = np.mean(((Y - Y_hat) / denom) ** 2, axis=0)

            coeffs = np.linalg.solve(R, Q.T @ Y)  # (deg + 1, n_spec)
            u_pows = np.vander(u_targ, N=deg + 1, increasing=True)
            pred = np.einsum("ij,ji->i", u_pows, coeffs)

            better = loo_mse < best_loo
            best_loo = np.where(better, loo_mse, best_loo)
            best_pred = np.where(better, pred, best_pred)

        return best_pred

    def get_sr(self) -> np.ndarray:
        """
        Return spectral response (SR)

        Returns:
            np.ndarray: spectral response data
        """
        # SR [A/W] = EQE / E_photon[eV]; wavelength_to_photonenergy converts nm -> eV
        sr = self.eqe.T / np.asarray(convert.wavelength_to_photonenergy(self.wavelength.flatten()))
        return sr.T

    def add_eqe(self, wavelength_add: np.ndarray, eqe_add: np.ndarray, temperature_add: float | np.ndarray = 25, sjuncs: str | None = None) -> None:  # ty: ignore[invalid-method-override]
        """
        Add EQE(T)

        Args:
            wavelength_add (np.ndarray[float]): wavelength of the new eqe
            eqe_add (np.ndarray[float]): new eqe data
            temperature_add (Union[float,np.ndarray[float]]): temperautre or array of temperatures where eqe/s was/were measured
            sjuncs (str, optional): name of the junction in case of multijunction devices. Defaults to None.V
        """
        # make array in case of float for np.concatenate
        temperature_add = np.atleast_1d(temperature_add) if not isinstance(temperature_add, np.ndarray) else temperature_add

        super().add_eqe(wavelength_add, eqe_add)
        self.temperature = np.concatenate((self.temperature, temperature_add))
        self._sort_by_temperature()

    def plot(self, fig: plt.Figure | None = None, ax: plt.Axes | None = None) -> tuple[plt.Axes, plt.Axes]:  # ty: ignore[invalid-method-override]
        """
        Plot the EQE(T). Lines are colored by temperature.

        Args:
            fig (plt.Figure, optional): Will use matplotlib Figure if provided. Defaults to None.
            ax (plt.Axes, optional): Will use matplotlib Axes if provided. Defaults to None.
        """

        if len(self.temperature) == 1 or np.allclose(self.temperature, self.temperature[0]):
            ax, rax = super().plot(fig=fig, ax=ax)
            return ax, rax

        # Check fig,ax and make new if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        if len(fig.get_axes()) > 1:
            fig.set_layout_engine("constrained")

        # prepare colormap
        nr_colorsegments = 256
        # cmap = mpl.colors.LinearSegmentedColormap.from_list("rg", ["b", "r"], N=nr_colorsegments)
        cmap = plt.get_cmap("coolwarm")

        min_temperature = self.temperature.min()
        max_temperature = self.temperature.max()

        # loop and plot
        for i, line in enumerate(self.eqe.T):
            color_idx = int((self.temperature[i] - min_temperature) / (max_temperature - min_temperature) * nr_colorsegments)
            ax.plot(self.wavelength, line, color=cmap(color_idx))

        # make colorbar
        color_norm = mpl.colors.Normalize(min_temperature, max_temperature)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=color_norm)
        fig.colorbar(sm, ax=ax, orientation="horizontal", location="top", pad=0.01, label="Temperature (degC)")

        # set labels
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("EQE")

        rax = ax.twinx()  # right axis
        return ax, rax

    def plot_sr(self, fig: plt.Figure | None = None, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the spectral response SR(T). Lines are colored by temperature.

        Args:
            fig (plt.Figure, optional): Will use matplotlib Figure if provided. Defaults to None.
            ax (plt.Axes, optional): Will use matplotlib Axes if provided. Defaults to None.
        """

        # Check fig,ax and make new if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        if len(fig.get_axes()) > 1:
            fig.set_layout_engine("constrained")

        # prepare colormap
        nr_colorsegments = 256
        # cmap = mpl.colors.LinearSegmentedColormap.from_list("rg", ["b", "r"], N=nr_colorsegments)
        cmap = plt.get_cmap("coolwarm")

        min_temperature = self.temperature.min()
        max_temperature = self.temperature.max()

        # calculate spectral response (SR)
        sr = self.get_sr().T

        # loop and plot
        for i, line in enumerate(sr):
            color_idx = int((self.temperature[i] - min_temperature) / (max_temperature - min_temperature) * nr_colorsegments)
            ax.plot(self.wavelength, line, color=cmap(color_idx))

        # make colorbar
        color_norm = mpl.colors.Normalize(min_temperature, max_temperature)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=color_norm)
        fig.colorbar(sm, ax=ax, orientation="horizontal", location="top", pad=0.01, label="Temperature (degC)")

        # set labels
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Spectral response (A W$^{-1}$)")
        return fig, ax


# Model functions for temperature dependents of bandgap and sigma


def _piecewise_linear_model(temperature: np.ndarray, t_split: float, m1: float, c1: float, m2: float, c2: float) -> np.ndarray:
    """Model function for the piecewise linear model."""
    # temperature = np.array(temperature)
    # res = np.zeros_like(temperature, dtype=np.float64)
    # t_filter = temperature >= t_split
    # # Above t_split
    # res[t_filter] = m1 * temperature[t_filter] + c1
    # # Above t_split
    # res[~t_filter] = m2 * t_split + c2

    temperature = np.array(temperature)
    res = np.zeros_like(temperature, dtype=np.float64)
    t_filter = temperature >= t_split
    # Above t_split
    res[t_filter] = m1 * temperature[t_filter] + c1
    # Below t_split
    c2 = (m1 - m2) * t_split + c1  # making piecewise-linear continous at t_split
    res[~t_filter] = m2 * temperature[~t_filter] + c2
    return res


def _linear_model(temperature: np.ndarray, p0: float, p1: float) -> np.ndarray:
    """Model function for the linear model."""
    return p0 * temperature + p1


def _polynomial_model(order: int) -> tuple[Callable, list[float]]:
    """
    Generates a polynomial model function of a specified order and an initial guess.

    Args:
        order (int): The order of the polynomial (e.g., 3 for cubic).

    Returns:
        Tuple[Callable, List[float]]: A polynomial model function and initial guess list.
    """

    def model(temperature, *params):
        return sum(params[i] * temperature ** (order - i) for i in range(order + 1))

    initial_guess = [0.0] * order + [1.0]  # Last coefficient set to 1.0 for constant term
    return model, initial_guess


def _poly2(temperature: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """2nd order polynomial model: a * T^2 + b * T + c"""
    return a * temperature**2 + b * temperature + c


def _poly3(temperature: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """3rd order polynomial model: a * T^3 + b * T^2 + c * T + d"""
    return a * temperature**3 + b * temperature**2 + c * temperature + d


def _poly4(temperature: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    """4th order polynomial model: a * T^4 + b * T^3 + c * T^2 + d * T + e"""
    return a * temperature**4 + b * temperature**3 + c * temperature**2 + d * temperature + e


def _poly5(temperature: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    """5th order polynomial model: a * T^5 + b * T^4 + c * T^3 + d * T^2 + e * T + f"""
    return a * temperature**5 + b * temperature**4 + c * temperature**3 + d * temperature**2 + e * temperature + f


def _phonon_bose_model(temperature: np.ndarray, sigma_0: float, sigma_ph: float, omega_ph: float) -> np.ndarray:
    r"""Bose-Einstein phonon-occupation broadening model.

    Returns:
        \sigma(T) = \sigma_0 + \sigma_ph / (exp(\hbar \omega_ph / kT) - 1)

    Parameters
    ----------
    sigma_0 : static disorder broadening (T-independent) [eV]
    sigma_ph : phonon coupling strength [eV]
    omega_ph : phonon energy [eV], constrained to [5, 100] meV.
        Typical values: perovskite ~15 meV, Si ~26 meV, GaAs ~36 meV.

    After normalisation by 'TemperatureModel.fit', 'sigma_0' and
    'sigma_ph' are scaled so the model evaluates to 1.0 at T_ref.

    Refs: Cody et al., Phys. Rev. Lett. 47, 1480 (1981);
    Ledinsky et al., J. Phys. Chem. Lett. 10, 1368 (2019).
    """
    temperature = np.asarray(temperature, dtype=float)
    TK_val = convert.TK(temperature)  # degC -> K
    kT = k_q * TK_val  # thermal voltage [eV] (k_q = k_B / q)
    return sigma_0 + sigma_ph / np.expm1(omega_ph / kT)


def _spline3(temperature: np.ndarray, *params: float) -> np.ndarray:
    temperature = np.asarray(temperature)
    x_vals = np.linspace(min(temperature), max(temperature), len(params))
    spline = UnivariateSpline(x_vals, params, k=3, s=0, ext=0)  # 0 for extrapolation
    return spline(temperature)


def _spline_model(spline_order: int) -> tuple[Callable, list[float]]:
    """
    Generates a spline model function of a specified order with extrapolation enabled.

    Args:
        spline_order (int): The order of the spline (e.g., 1 for linear, 3 for cubic).

    Returns:
        Tuple[Callable, List[float]]: A spline model function and initial guess list.
    """

    def model(temperature, *params):
        temperature = np.asarray(temperature)
        x_vals = np.linspace(min(temperature), max(temperature), len(params))
        spline = UnivariateSpline(x_vals, params, k=spline_order, s=0, ext=3)  # `ext=3` for extrapolation
        return spline(temperature)

    initial_guess = [1.0] * 5  # Example initial guess with 5 control points
    return model, initial_guess


class ModelType(Enum):
    LINEAR = ("linear", _linear_model, [1.0, 0.0])  # [slope, intercept]
    PIECEWISE_LINEAR = ("piecewise_linear", _piecewise_linear_model, [120.0, 1.0, 0.0, 1.0, 0.0])  # [t_split, m1, c1, m2, c2]
    POLY2 = ("2nd_order_polynomial", _poly2, [1.0, 0.0, 0.0])
    POLY3 = ("3rd_order_polynomial", _poly3, [1.0, 0.0, 0.0, 0.0])
    POLY4 = ("4th_order_polynomial", _poly4, [1.0, 0.0, 0.0, 0.0, 0.0])
    POLY5 = ("5th_order_polynomial", _poly5, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    PHONON_BOSE = ("phonon_bose", _phonon_bose_model, [0.5, 0.5, 0.015], ([0.0, 0.0, 0.005], [np.inf, np.inf, 0.100]))  # [sigma_0, sigma_ph, omega_ph]
    # SPLINE3 = ("cubic_spline", _spline3, [1.0] * 5)

    def __init__(self, name: str, function: Callable, initial_guess: list[float], bounds=None):
        self._function = function
        self._initial_guess = list(initial_guess)
        self._bounds = bounds  # (lower, upper) for least_squares, or None for unconstrained

    @property
    def function(self) -> Callable:
        return self._function

    @property
    def initial_guess(self) -> list[float]:
        return list(self._initial_guess)

    @property
    def bounds(self):
        """Parameter bounds for least_squares: (lower, upper) arrays or (-inf, inf)."""
        if self._bounds is not None:
            return (list(self._bounds[0]), list(self._bounds[1]))
        n = len(self._initial_guess)
        return ([-np.inf] * n, [np.inf] * n)


def _model_residuals(params: tuple[float, ...], func: Callable, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Residual calculation for least squares
    Args:
        params (Tuple): params for the fit function
        func (Callable): fit function
        x (np.ndarray): x data
        y (np.ndarray): y data

    Returns:
        np.ndarray: array with residuals
    """

    # center = np.mean(x)
    # sigma = (max(x) - min(x)) / 6
    # weights = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    # weights_start = np.exp(-0.5 * ((x - min(x)) / sigma) ** 2)
    # weights_end = np.exp(-0.5 * ((x - max(x)) / sigma) ** 2)
    # weights = weights_start + weights_end
    weights = 1
    predicted = func(x, *params)
    return weights * (y - predicted).ravel()


def _calc_mic(temperature: np.ndarray, predicted: np.ndarray, actual: np.ndarray, params: Sequence[float]) -> tuple[float, float]:
    """Calculate small-sample corrected AIC (AICc) and MSE.

    Returns 'inf' for models with 'n - k - 1 <= 0' (exact/near-exact fits) so
    saturated polynomials cannot win model selection via log(0).
    """
    n = int(len(temperature))
    k = int(len(params))
    mse = float(np.mean((predicted - actual) ** 2))
    if n - k - 1 <= 0:
        return float("inf"), mse
    sigma2 = max(mse, float(np.finfo(float).tiny))
    aicc = n * np.log(sigma2) + 2.0 * k + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    return float(aicc), mse


_POLY_DEGREES = {
    ModelType.LINEAR: 1,
    ModelType.POLY2: 2,
    ModelType.POLY3: 3,
    ModelType.POLY4: 4,
    ModelType.POLY5: 5,
}


def _fit_centered_poly_in_raw_T(x: np.ndarray, y: np.ndarray, deg: int) -> tuple[np.ndarray, float]:
    """Fit a degree-'deg' polynomial in centered/scaled T and return raw-T coefficients + LOO MSE."""
    x_mean = float(x.mean())
    x_scale = max(float(x.std()), 1.0)
    u = (x - x_mean) / x_scale
    V = np.vander(u, N=deg + 1, increasing=False)  # highest power first
    Q, R = np.linalg.qr(V, mode="reduced")
    if np.linalg.matrix_rank(R) < deg + 1:
        raise RuntimeError("Rank-deficient polynomial Vandermonde matrix")
    c_u = np.linalg.solve(R, Q.T @ y)
    y_hat = Q @ (Q.T @ y)
    if deg + 1 >= x.size:
        loo_mse = float("inf") if x.size > 2 else float(np.mean((y - y_hat) ** 2))
    else:
        h = np.sum(Q * Q, axis=1)
        loo_mse = float(np.mean(((y - y_hat) / np.maximum(1.0 - h, 1e-12)) ** 2))

    # Convert p_u((T - x_mean) / x_scale) to raw-T polynomial coefficients
    poly_raw = np.poly1d([0.0])
    shift_poly = np.poly1d([1.0 / x_scale, -x_mean / x_scale])
    for coeff in c_u:
        poly_raw = poly_raw * shift_poly + float(coeff)
    raw_coeffs = np.zeros(deg + 1, dtype=np.float64)
    raw_coeffs[-(len(poly_raw.coeffs)) :] = poly_raw.coeffs
    return raw_coeffs, loo_mse


def _fit_nonlinear_model(model_type: ModelType, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """least_squares fit of a non-polynomial ModelType (PIECEWISE_LINEAR, PHONON_BOSE) to (x, y)."""
    init = model_type.initial_guess
    b_lo, b_hi = model_type.bounds
    if model_type is ModelType.PIECEWISE_LINEAR:
        span = float(x.max() - x.min())
        init[0] = float(np.median(x))
        slope0 = float((y[-1] - y[0]) / span) if span > 0 else 0.0
        c0 = float(y.mean() - slope0 * x.mean())
        init[1:] = [slope0, c0, slope0, c0]
        if span > 0:
            b_lo[0] = float(x.min() + 0.1 * span)
            b_hi[0] = float(x.max() - 0.1 * span)
            init[0] = min(max(init[0], b_lo[0]), b_hi[0])
    elif model_type is ModelType.PHONON_BOSE:
        init[0] = max(float(np.min(y)) * 0.8, 0.0)
        init[1] = max(float(np.max(y) - np.min(y)), 1e-4)
    result = least_squares(_model_residuals, init, args=(model_type.function, x, y), loss="soft_l1", bounds=(b_lo, b_hi))
    return result.x.copy()


def _loo_mse_nonlinear(model_type: ModelType, x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out cross-validation MSE [y units^2] of a non-polynomial ModelType, one refit per point.

    The same criterion as the closed-form LOO MSE of _fit_centered_poly_in_raw_T,
    so polynomial and non-polynomial candidates can be compared.
    """
    errors = np.empty(x.size, dtype=np.float64)
    for i in range(x.size):
        keep = np.arange(x.size) != i
        params = _fit_nonlinear_model(model_type, x[keep], y[keep])
        errors[i] = float(np.ravel(model_type.function(x[i : i + 1], *params))[0]) - float(y[i])
    return float(np.mean(errors**2))


class TemperatureModel:
    """
    Temperature model for bandgap and sigma
    """

    def __init__(
        self,
        model_type: ModelType,
        params: list[float] | np.ndarray,
        Tref: float = 25,
        ref_value: float | None = None,
        t_range: tuple[float, float] | None = None,
    ):
        """
        Initializes the model with a specified type and parameters.

        Args:
            model_type (ModelType): Type of temperature dependence model.
            params (List[float]): Parameters for the chosen model, normalised so that
                'model(Tref) == 1'.
            Tref (float): Reference temperature [degC].
            ref_value (float, optional): Value of the fitted quantity at 'Tref'
                (e.g. Eg(25 degC) in eV). Set automatically by 'fit()'; used as
                the default scale in 'apply()'.
            t_range (Tuple[float, float], optional): Measured temperature range
                '(T_min, T_max)' [degC] used during 'fit()'.
        """
        if model_type not in ModelType:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        self.model_type = model_type
        self.params = params
        self.Tref = Tref
        self.ref_value = ref_value
        self.t_range = t_range

    def apply(self, temperature: float | np.ndarray, ref_value: float | None = None) -> float | np.ndarray:
        r"""
        Applies the specified model to temperature and scales it with a reference value.

        Args:
            temperature (Union[float, np.ndarray]): Target temperature [degC].
            ref_value (float, optional): Value at 'Tref'. Defaults to 'self.ref_value'.

        Returns:
            Union[float, np.ndarray]: Calculated value at target temperature.
        """
        if ref_value is None:
            ref_value = self.ref_value
        if ref_value is None:
            raise ValueError("TemperatureModel.apply: no ref_value given and none stored from fit(); pass ref_value explicitly.")
        temp_arr = np.asarray(temperature, dtype=np.float64)
        if self.t_range is not None:
            fin = temp_arr[np.isfinite(temp_arr)]
            t_lo, t_hi = self.t_range
            if fin.size and (np.any(fin < t_lo) or np.any(fin > t_hi)):
                logger.warning(
                    "TemperatureModel.apply ({}): extrapolating to [{:.1f}, {:.1f}] C outside fitted range [{:.1f}, {:.1f}] C",
                    self.model_type.name,
                    float(fin.min()),
                    float(fin.max()),
                    t_lo,
                    t_hi,
                )
        model_func = self.model_type.function
        out = model_func(temp_arr, *self.params) * ref_value
        return float(out) if np.ndim(temperature) == 0 else out

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        plot: str | None = None,
        model_types: ModelType | list[ModelType] | None = None,
        Tref: float = 25,
    ) -> TemperatureModel:
        """Fit temperature-dependence model(s) to (x, y) and return the best TemperatureModel.

        Defaults to '[ModelType.LINEAR]' when 'model_types=None' to avoid
        polynomial overfitting on sparse laboratory temperature series. When
        multiple 'model_types' are supplied, every candidate is scored by its
        leave-one-out cross-validation MSE (closed form for the polynomials,
        one refit per point for PIECEWISE_LINEAR and PHONON_BOSE, which need
        at least 4 distinct temperatures), and saturated polynomials are
        excluded.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        if x.size != y.size or x.size < 2:
            raise ValueError(f"TemperatureModel.fit requires matching x and y of length >= 2, got {x.size} and {y.size}")

        best_score = float("inf")
        best_params = None
        best_model_type = None

        if plot is not None:
            _, ax = plt.subplots(1, 1, layout="constrained")
            ax.plot(x, y, "*", label="data")
            ax.set_xlabel("Temperature (degC)")
            ax.set_ylabel("Bandgap/sigma (eV)")

        if model_types is None:
            candidates = [ModelType.LINEAR]
        else:
            candidates = list(model_types) if isinstance(model_types, list) else [model_types]

        multi = len(candidates) > 1
        x_unique = np.unique(x)
        n_u = x_unique.size

        for model_type in candidates:
            func = model_type.function
            try:
                if model_type in _POLY_DEGREES:
                    deg = _POLY_DEGREES[model_type]
                    if deg + 1 > n_u:
                        continue
                    if multi and deg > max(1, n_u - 2):
                        continue
                    params, loo_mse = _fit_centered_poly_in_raw_T(x, y, deg)
                    predicted = func(x, *params)
                    mic, mse = _calc_mic(x, predicted, y, params)
                    score = loo_mse if multi else (mic if math.isfinite(mic) else mse)
                else:
                    if multi and n_u < 4:
                        continue
                    params = _fit_nonlinear_model(model_type, x, y)
                    predicted = func(x, *params)
                    _, mse = _calc_mic(x, predicted, y, params)
                    if multi:
                        try:
                            score = _loo_mse_nonlinear(model_type, x, y)
                        except (RuntimeError, ValueError, np.linalg.LinAlgError):
                            score = float("inf")
                    else:
                        score = mse

                if score < best_score or best_model_type is None:
                    best_score = score
                    best_params = params.copy()
                    best_model_type = model_type

                if plot == "all":
                    t_space = np.linspace(x.min(), x.max())
                    pred = func(t_space, *params)
                    ax.plot(t_space, pred, "--", label=f"{model_type.name} score={score:.2e} MSE={mse:.2e}")

            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue

        if best_model_type is None or best_params is None:
            raise ValueError("No model could be fit to the data.")

        if plot == "best":
            func = best_model_type.function
            t_space = np.linspace(x.min(), x.max())
            pred = func(t_space, *best_params)
            ax.plot(t_space, pred, "--", label=best_model_type.name)

        if plot is not None:
            ax.legend(fontsize=8)

        ref_value = float(best_model_type.function(Tref, *best_params))

        # normalize params to ref value (on a copy, never mutating best_params)
        normalized_params = best_params.copy()
        if best_model_type is ModelType.PIECEWISE_LINEAR:
            normalized_params[1:] /= ref_value
        elif best_model_type is ModelType.PHONON_BOSE:
            # omega_ph (index 2) is a physical energy [eV] and must NOT be scaled by ref_value
            normalized_params[0] /= ref_value
            normalized_params[1] /= ref_value
        else:
            normalized_params = normalized_params / ref_value

        return cls(
            best_model_type,
            normalized_params,
            Tref=Tref,
            ref_value=ref_value,
            t_range=(float(x.min()), float(x.max())),
        )
