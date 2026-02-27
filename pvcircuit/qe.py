# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package.
    pvcircuit.qe    # functions for QE analysis
"""
from __future__ import annotations

import math  # simple math
from enum import Enum
from functools import lru_cache
from pathlib import Path
from time import time
from typing import Callable, Dict, List, Tuple, Union

import matplotlib as mpl  # plotting
import matplotlib.pyplot as plt  # plotting
import numpy as np  # arrays
import pandas as pd  # dataframes

# from scipy.integrate import trapezoid
from scipy import constants  # physical constants
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline, interp1d
from scipy.optimize import (
    brentq,  # root finder
    curve_fit,
    fsolve,
    least_squares,
)
from scipy.special import erfc, gamma, gammaincc, lambertw  # special functions

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
hc_k = constants.h * constants.c / constants.k * 1e9  # for wavelength (nm)
DB_PREFIX = 2.0 * np.pi * constants.e * (constants.k / constants.h) ** 3 / (constants.c) ** 2 / 1.0e4  # about 1.0133e-8 for Jdb[A/cm2]
nm2eV = constants.h * constants.c / constants.e * 1e9
JCONST = 1000 / 100 / 100 / nm2eV  # mA/cm2
DBWVL_PREFIX = 2.0 * np.pi * constants.c * constants.e / 100 / 100  # A/cm2


# standard spectra
ASTMfile = pvc.datapath.joinpath("ASTMG173.csv")

try:
    dfrefspec = pd.read_csv(ASTMfile, index_col=0, header=2)
    wvl = dfrefspec.index.to_numpy(dtype=np.float64, copy=True)
    refspec = dfrefspec.to_numpy(dtype=np.float64, copy=True)  # all three reference spectra
    refnames = ["space", "global", "direct"]
    AM0 = refspec[:, 0]  # dfrefspec['space'].to_numpy(dtype=np.float64, copy=True)  # 1348.0 W/m2
    AM15G = refspec[:, 1]  # dfrefspec['global'].to_numpy(dtype=np.float64, copy=True) # 1000.5 W/m2
    AM15D = refspec[:, 2]  # dfrefspec['direct'].to_numpy(dtype=np.float64, copy=True) # 900.2 W/m2
except:
    print(pvc.pvcpath)
    print(pvc.datapath)
    print(ASTMfile)


def ordinal(n: int) -> str:
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = suffixes.get(n % 10, "th")
    return f"{n}{suffix}"


def _eq_solve_Eg(Eg: float, *data: Tuple[np.ndarray, np.ndarray]) -> float:
    x, y = data
    return trapezoid(x * y, x) / trapezoid(y, x) - Eg


def _gaussian(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    return 1 * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def JdbMD(EQE: Union[np.ndarray, List[float]], xEQE: Union[np.ndarray, List[float]], TC: float, Eguess: float = 1.0, kTfilter: int = 3, bplot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    calculate detailed-balance reverse saturation current
    from EQE vs xEQE
    xEQE in nm, can optionally use (start, step) for equally spaced data
    debug on bplot
    """
    Vthlocal = convert.Vth(TC)  # kT
    EQE = np.array(EQE)  # ensure numpy
    if EQE.ndim == 1:  # 1D EQE[lambda]
        (nQlams,) = EQE.shape
        njuncs = 1
    elif EQE.ndim == 2:  # 2D EQE[lambda, junction]
        nQlams, njuncs = EQE.shape
    else:
        return "dims in EQE:" + str(EQE.ndim)

    Eguess = np.array([Eguess] * njuncs)

    if len(xEQE) == 2:  # evenly spaced x-values (start, stop)
        start, stop = xEQE
        # stop =  start + step * (nQlams - 1)
        # xEQE, step = np.linspace(start, stop, nQlams, dtype=np.float64, retstep=True)
        xEQE = np.linspace(start, stop, nQlams, dtype=np.float64)
    else:  # arbitrarily spaced x-values
        xEQE = np.array(xEQE, dtype=np.float64)
        start = min(xEQE)
        stop = max(xEQE)
        # step = xEQE[1]-xEQE[0]  #first step

    if xEQE.ndim != 1:  # need 1D with same length as EQE(lam)
        return "dims in xEQE:" + str(xEQE.ndim) + "!=1"
    elif len(xEQE) != nQlams:
        return "nQlams:" + str(len(xEQE)) + "!=" + str(nQlams)

    Egvect = np.vectorize(EgFromJdb)
    EkT = nm2eV / Vthlocal / xEQE
    blackbody = np.expand_dims(DBWVL_PREFIX / (xEQE * 1e-9) ** 4 / np.expm1(EkT), axis=1)

    for count in range(10):
        nmfilter = nm2eV / (Eguess - Vthlocal * kTfilter)  # MD [652., 930.]
        if njuncs == 1:
            EQEfilter = np.expand_dims(EQE.copy(), axis=1)
        else:
            EQEfilter = EQE.copy()

        for i, lam in enumerate(xEQE):
            EQEfilter[i, :] *= lam < nmfilter  # zero EQE about nmfilter

        DBintegral = blackbody * EQEfilter
        Jdb = trapezoid(DBintegral, x=(xEQE * 1e-9), axis=0)
        Egnew = Egvect(TC, Jdb)
        if bplot:
            print(Egnew, max((Egnew - Eguess) / Egnew))
        if np.amax((Egnew - Eguess) / Egnew) < 1e-6:
            break
        else:
            Eguess = Egnew

    if bplot:
        efig, eax = plt.subplots()
        eax.plot(xEQE, DBintegral[:, 0], c="blue", lw=2, marker=".")
        if njuncs > 1:
            reax = eax.twinx()  # right axis
            reax.plot(xEQE, DBintegral[:, 1], c="red", lw=2, marker=".")

    return Jdb, Egnew


def PintMD(Pspec: Union[str, np.ndarray], xspec: np.ndarray = wvl) -> np.ndarray:
    # optical power of spectrum over full range
    return JintMD(None, None, Pspec, xspec)


def JintMD(EQE: Union[np.ndarray, None], xEQE: Union[np.ndarray, List[float], None], Pspec: Union[str, np.ndarray], xspec: np.ndarray = wvl) -> np.ndarray:
    """
    integrate over spectrum or spectra
    if EQE is None -> calculate Power in [W/m2]
        total power=int(Pspec) over xEQE range
    else             -> caluculate J = spectra * lambda * EQE(lambda)
        Jsc = int(Pspec*QE[0]*lambda) in [mA/cm2]
        EQE optionally scalar for constant over xEQE range
        integrate over full spectrum is xEQE is None

    integrate multidimentional EQE(lambda)(junction) times MD reference spectra Pspec(lambda)(ispec)
    external quantum efficiency EQE[unitless] x-units = nm,
    reference spectra Pspec[W/m2/nm] x-units = nm
    optionally Pspec as string 'space', 'global', 'direct' or '' for all three
    xEQE in nm, can optionally
        use (start, step) for equally spaced data
        use None for same as xspec
    default x values for Pspec from wvl
    """

    # check spectra input
    if type(Pspec) is str:  # optional string space, global, direct
        if Pspec in dfrefspec.columns:
            Pspec = dfrefspec[Pspec].to_numpy(dtype=np.float64, copy=True)
        else:
            Pspec = dfrefspec.to_numpy(dtype=np.float64, copy=True)  # just use refspec instead of error
    else:
        Pspec = np.array(Pspec, dtype=np.float64)  # ensure numpy

    if Pspec.ndim == 1:  # 1D Pspec[lambda]
        (nSlams,) = Pspec.shape
        nspecs = 1
    elif Pspec.ndim == 2:  # 2D Pspec[lambda, ispec]
        nSlams, nspecs = Pspec.shape
    else:
        return "dims in Pspec:" + str(Pspec.ndim)

    # check EQE input
    EQE = np.array(EQE)  # ensure numpy
    if EQE.ndim == 0:  # scalar or None
        if np.any(EQE):
            nQlams = 1  # scalar -> return current
            njuncs = 1
        else:
            nQlams = 1  # None or False -> return power
            njuncs = 0
    elif EQE.ndim == 1:  # 1D EQE[lambda]
        (nQlams,) = EQE.shape
        njuncs = 1
    elif EQE.ndim == 2:  # 2D EQE[lambda, junction]
        nQlams, njuncs = EQE.shape
    else:
        return "dims in EQE:" + str(EQE.ndim)

    # check x range input
    xEQE = np.array(xEQE, dtype=np.float64)
    if xEQE.ndim == 0:  # scalar or None
        xEQE = xspec  # use spec range if no EQE range
    if xEQE.ndim == 1:  # 1D
        (nxEQE,) = xEQE.shape
        start = min(xEQE)
        stop = max(xEQE)
        if nxEQE == 2:  # evenly spaced x-values (start, stop)
            xEQE = np.linspace(start, stop, max(nQlams, 2), dtype=np.float64)
    else:
        return "dims in xEQE:" + str(xEQE.ndim)

    if nQlams == 1:
        EQE = np.full_like(xEQE, EQE)

    if xspec.ndim != 1:  # need 1D with same length as Pspec(lam)
        return "dims in xspec:" + str(xspec.ndim) + "!=1"
    elif len(xspec) != nSlams:
        return "nSlams:" + str(len(xspec.ndim)) + "!=" + str(nSlams)

    if xEQE.ndim != 1:  # need 1D with same length as EQE(lam)
        return "dims in xEQE:" + str(xEQE.ndim) + "!=1"
    elif nQlams == 1:
        pass
    elif len(xEQE) != nQlams:
        return "nQlams:" + str(len(xEQE)) + "!=" + str(nQlams)

    # find start and stop index  of nSlams
    n0 = 0
    n1 = nSlams - 1
    for i, lam in enumerate(xspec):
        if lam <= min(start, stop):
            n0 = i
        elif lam <= max(start, stop):
            n1 = i
        else:
            break
    xrange = xspec[n0 : n1 + 1]  # range of xspec values within xEQE range
    nrange = abs(n1 + 1 - n0)
    if njuncs == 0:  # calculate power over xrange
        if nspecs == 1:
            Jintegral = Pspec.copy()[n0 : n1 + 1]  # for Ptot
        else:
            Jintegral = Pspec.copy()[n0 : n1 + 1, :]  # for Ptot

    else:  # calculate J over xrange
        # print(xrange.shape, xEQE.shape, EQE.shape, Pspec.shape)
        EQEinterp = interp1d(xEQE, EQE, axis=0, fill_value=0)  # interpolate along axis=0
        Jintegral = np.zeros((nrange, nspecs, njuncs), dtype=np.float64)  # 3D array
        if njuncs == 1:
            EQEfine = np.expand_dims((EQEinterp(xrange) * xrange), axis=1) * JCONST  # lambda*EQE(lambda)[lambda,1]
        else:
            EQEfine = EQEinterp(xrange) * xrange[:, np.newaxis] * JCONST  # lambda*EQE(lambda)[lambda,junc]
        for ijunc in range(0, njuncs):
            if nspecs == 1:
                Jintegral[:, 0, ijunc] = Pspec.copy()[n0 : n1 + 1]  # for Ptot
            else:
                Jintegral[:, :, ijunc] = Pspec.copy()[n0 : n1 + 1, :]  # for Ptot
            Jintegral[:, :, ijunc] *= EQEfine[:, np.newaxis, ijunc]

    Jint = trapezoid(Jintegral, x=xrange, axis=0)
    # print(xrange.shape, EQEfine.shape, EQE.shape, Pspec.shape, Jintegral.shape)
    # print(nSlams, nspecs, njuncs, nQlams, start, stop, xspec[n0], xspec[n1], n0, n1)
    return Jint


@lru_cache(maxsize=100)
def JdbFromEg(TC: float, Eg: float, dbsides: float = 1.0, method: str = None) -> float:
    """
    return the detailed balance dark current
    assuming a square EQE
    Eg[=]eV
    TK[=]K
    returns Jdb[=]A/cm2

    optional parameters
    method: 'gamma'
    dbsides:    single-sided->1.  bifacial->2.
    see special functions, Andrews p.71
    see Geisz EL&LC paper, King EUPVSEC
    """
    EgkT = Eg / convert.Vth(TC)
    TKlocal = convert.TK(TC)

    if str(method).lower == "gamma":
        # use special function incomplete gamma
        # gamma(3)=2.0 not same incomplete gamma as in Igor
        Jdb = DB_PREFIX * TKlocal**3.0 * gammaincc(3.0, EgkT) * 2.0 * dbsides
    else:
        # Jdb as in Geisz et al.
        Jdb = DB_PREFIX * TKlocal**3.0 * (EgkT * EgkT + 2.0 * EgkT + 2.0) * np.exp(-EgkT) * dbsides  # units from DB_PREFIX

    return Jdb


@lru_cache(maxsize=100)
def EgFromJdb(TC: float, Jdb: float, Eg: float = 1.0, eps: float = 1e-6, itermax: int = 100, dbsides: float = 1.0) -> Union[float, None]:
    """
    see GetData AT_Egcalc
    return the bandgap from the Jdb
    assuming a square EQE
    iterates using gammaInc(3,x)=2*exp(-x)*(1+x+x^2/2)
    see special functions, Andrews p.73

    optional parameters
    Eg=1.0 eV    #initial guess
    eps=0.001    #tolerance
    itermax=100 #maximum iterations
    dbsides=1.  #bifacial->2.
    """

    Vthlocal = convert.Vth(TC)
    TKlocal = convert.TK(TC)
    x0 = Eg / Vthlocal
    off = np.log(2.0 * dbsides * DB_PREFIX * TKlocal**3.0 / Jdb)
    count = 0

    while count < itermax:
        x1 = off + np.log(1.0 + x0 + x0 * x0 / 2.0)
        try:
            tol = abs((x1 - x0) / x0)
        except:
            tol = abs(x1 - x0)
        if tol < eps:
            return x1 * Vthlocal
        x0 = x1
        count += 1
    return None


def ensure_numpy_2drow(array: Union[np.ndarray, List[float]]) -> np.ndarray:

    # ensure numpy
    array = np.array(array)
    # ensure 2D
    array = array.reshape(1, -1) if array.ndim == 1 else array
    return array


def ensure_numpy_2dcol(array: Union[np.ndarray, List[float]]) -> np.ndarray:

    # ensure numpy
    array = np.array(array)
    # ensure 2D
    array = array.reshape(-1, 1) if array.ndim == 1 else array
    return array


class EQE(object):
    """
    EQE object
    It creates a class containing nth junctions EQEs and Luminescent Coupling
    between junctions. The contribution can be studied interectively using
    ipywidgets under a notebook.
    NOTE: Wavelengths become an N×1 array, and EQE becomes an N×M array
    (N is the number of wavelength points, and M is the number of junctions).
    Each row corresponds to a single wavelength entry; columns represent
    different junctions.

    The methods
    self.control create the self.ui to adjust the LC
    self.plot function to plot
    sle.Jdb: it gets the reverse saturation current.

    """

    def __init__(self, wavelength: np.ndarray, eqe: np.ndarray, name: str = "EQE", sjuncs: Union[List[str], None] = None):
        """It creats the EQE class. ntegrate over spectrum or spectra
        rawEQE (numpy.array):  2D(lambda)(junction) raw input rawEQE (not LC corrected)
        xEQE(numpy.array)      xEQE        # wavelengths [nm] for rawEQE data
        name (str):            name of EQE object, sample
        sjuncs [list of str]:          labels for the junctions, if None it is self generated

        The number of junctions is created from the dimension of the EQE
        self.njuncs = njuncs    # number of junctions
        """
        # Number of wavelength and EQE values must match
        if wavelength.shape[0] != eqe.shape[0]:
            raise ValueError(f"shape of wavelength and eqe doesn't match: {wavelength.shape[0]} != {eqe.shape[0]}")

        self.wavelength = ensure_numpy_2dcol(wavelength)
        self.eqe = ensure_numpy_2dcol(eqe)

        self.name = name

        self.njuncs = self.eqe.shape[1]  # number of junction

        if sjuncs == None:
            self.sjuncs = [ordinal(junc + 1) for junc in range(self.njuncs)]
        else:
            self.sjuncs = sjuncs  # names of junctions

        self.corrEQE = np.empty_like(self.eqe)  # luminescent coupling corrected EQE same size as rawEQE
        self.etas = np.zeros((self.njuncs, 3), dtype=np.float64)  # LC factor for next junctions
        self.LCcorr()  # calculate LC with zero etas
        self.spectra = None

    def add_spectra(self, wavelength: np.ndarray = None, spectra: np.ndarray = None) -> None:
        """
        Add spectral data

        Args:
            spectra (optional): array with wavelength and spectral data in W/m^2/nm. Defaults to AM1.5G.
        """

        if wavelength is None and spectra is None:
            wavelength = wvl
            spectra = AM15G

        if wavelength is None or spectra is None:
            raise ValueError("Provide wavelength and spectra")

        if wavelength.shape[0] != spectra.shape[0]:
            raise ValueError("Wavelength and Spectra need to be column vectors")

        # ensure numpy and 2D
        wavelength = np.array(wavelength)
        spectra = ensure_numpy_2dcol(spectra)

        # interpolate spectra to eqe
        f_spec_interp = interp1d(wavelength, spectra, kind="linear", axis=0)
        spectra_interp = f_spec_interp(self.wavelength.flatten())
        self.spectra = spectra_interp

    def add_eqe(self, wavelength_add: np.ndarray, eqe_add: np.ndarray, sjuncs: Union[str, None] = None) -> None:
        """
        Add eqe. Merge wavelength, inpolate eqe and fill extrapolation with 0. Assumes that the all EQE are part of a multijunction device.

        Args:
            eqe (optional): Wavelength of the new EQE
            eqe (optional): New EQE data
        """
        # ensure numpy
        wavelength_add = ensure_numpy_2dcol(wavelength_add)
        eqe_add = ensure_numpy_2dcol(eqe_add)

        # Combine old and new wavelength
        combined_wavelength = np.sort(np.unique(np.concatenate((self.wavelength, wavelength_add))))

        # Create interpolation functions with fill_value=0 for extrapolation
        f_interp_eqe = interp1d(self.wavelength.flatten(), self.eqe, kind="linear", fill_value=0, bounds_error=False, axis=0)
        f_interp_eqe_add = interp1d(wavelength_add.flatten(), eqe_add, kind="linear", fill_value=0, bounds_error=False, axis=0)

        # Interpolate and fill missing values
        eqe_interp = f_interp_eqe(combined_wavelength)
        eqe_add_interp = f_interp_eqe_add(combined_wavelength)

        # Assign new values
        self.wavelength = ensure_numpy_2dcol(combined_wavelength)
        self.eqe = np.hstack([eqe_interp, eqe_add_interp])

        self.njuncs = self.eqe.shape[1]  # number of junction

        if sjuncs is None:
            self.sjuncs.append(ordinal(self.njuncs))
        else:
            self.sjuncs.append(sjuncs)

        self.corrEQE = np.empty_like(self.eqe)  # luminescent coupling corrected EQE same size as rawEQE
        self.etas = np.zeros((self.njuncs, 3), dtype=np.float64)  # LC factor for next junctions
        self.LCcorr()  # calculate LC with zero etas

    def calc_Eg_Rau(self, return_sigma: bool = True, fit_gaussian: bool = True, plot_fits: bool = False) -> Tuple[List[float], List[float]]:
        # using U. Rau, B. Blank, T. C. M. Müller, and T. Kirchartz
        # 'Efficiency Potential of Photovoltaic Materials and Devices Unveiled by Detailed-Balance Analysis',
        # Phys. Rev. Applied, vol. 7, no. 4, p. 044016, Apr. 2017, doi: 10.1103/PhysRevApplied.7.044016.
        # extended by gaussian fit

        bandgaps = []
        sigmas = []
        # Define the Gaussian function
        for i in range(self.eqe.shape[1]):
            y = self.eqe[:, i]
            x = convert.wavelength_to_photonenergy(self.wavelength.flatten())

            # convert wavelength to photon energy
            y_grad = -1 * np.gradient(y)

            # filter tail to avoid eqe dips at end/beginning of measurement
            # y_grad = y_grad[(x < x[len(x) // 2])]
            # x = x[(x < x[len(x) // 2])]
            # data_filter = x < (x.max() + x.min()) / 2
            # x = x[data_filter]
            # y = y[data_filter]
            # y_grad = y_grad[data_filter]

            # we only need declining EQE to determine bandgaps
            data_filter = y_grad > 0
            x = x[data_filter]
            y = y[data_filter]
            y_grad = y_grad[data_filter]

            # normalize data
            y_grad = convert.normalize(y_grad)
            # get the index of the maximum
            y_diff_max_idx = np.nanargmax(y_grad)
            # get the max coordinates
            x_diff_max = x[y_diff_max_idx]
            y_diff_max = y_grad[y_diff_max_idx]

            # define lower threshold
            p_ab = np.exp(-2) * y_diff_max
            # thres = 0.5
            # p_ab = thres * y_diff_max
            # find the index of the low-energy side where P(a) is max(P(Eg)/2)
            a_cond = np.where((y_grad < p_ab) & (x < x_diff_max))[0]
            if len(a_cond > 0):
                a_idx = np.nanmin(a_cond)
            else:
                a_idx = len(x) - 1
            a = x[a_idx]
            p_a = y_grad[a_idx]
            # find the index of the high-energy side where P(b) is max(P(Eg)/2)
            b_idx = np.nanmax(np.where((y_grad < p_ab) & (x > x_diff_max))[0])
            b = x[b_idx]
            p_b = y_grad[b_idx]

            x_target = x[a_idx : b_idx - 1 : -1]
            y_target = y_grad[a_idx : b_idx - 1 : -1]

            if fit_gaussian:

                # initial guesses from weighted arithmetic mean and weighted sample sigma
                mean = sum(x_target * y_target) / sum(y_target)
                sigma = np.sqrt(sum(y_target * (x_target - mean) ** 2) / sum(y_target))

                fit_res = curve_fit(_gaussian, x_target, y_target, p0=[max(y_target), mean, sigma], method="trf")
                x_fit = np.linspace(x[b_idx], x[a_idx], 100)
                y_fit = _gaussian(x_fit, *fit_res[0])

                if plot_fits:
                    fig, ax = plt.subplots(1, 2, layout="constrained")
                    ax[0].plot(convert.wavelength_to_photonenergy(self.wavelength.flatten()), self.eqe[:, i])
                    ax[0].plot(x, y_grad, "--")
                    ax[1].plot(x, y_grad, "--")
                    ax[1].plot(x_fit, y_fit)
                    ax[1].plot(x_target, y_target, ".r")
                    ax[1].plot(x_diff_max, y_diff_max, "r*")
                    ax[1].plot(a, p_a, "g*")
                    ax[1].plot(b, p_b, "b*")

                    ax[0].set_xlabel(r"Photon energy $E_{\nu}$ [eV]")
                    ax[0].set_ylabel(r"EQE")
                    ax[1].set_xlabel(r"Photon energy $E_{\nu}$ [eV]")
                    ax[1].set_ylabel(r"$\frac{{\mathrm{d}}EQE}{{\mathrm{d}}E_{\nu}}$")

            else:
                fit_res = [None]
                x_fit = x_target
                y_fit = y_target
                sigma = None

            res = fsolve(_eq_solve_Eg, 1.0, args=(x_fit, y_fit))
            bandgap = res[0]

            bandgaps.append(bandgap)
            sigmas.append(sigma)
        return bandgaps, sigmas

    def LCcorr(self, junc: Union[int, None] = None, dist: Union[int, None] = None, val: Union[float, None] = None) -> None:
        """
        Applies the correction of the Luminescent coupling to the QE junc using procedure
        from Steiner et al., IEEE PV, v3, p879 (2013)
        """
        # change one eta[junc,dist] value
        # calculate LC corrected EQE
        etas = self.etas
        # with self.debugout: print(junc,dist,val)
        if junc == None or dist == None or val == None:
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

    def Jdb(self, TC: float, Eguess: float = 1.0, kTfilter: int = 3, dbug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """It calculate Jscs and Egs from self.corrEQE"""
        Vthlocal = convert.Vth(TC)  # kT
        Eguess = np.array([Eguess] * self.njuncs)
        Egvect = np.vectorize(EgFromJdb)
        EkT = nm2eV / Vthlocal / self.wavelength
        blackbody = DBWVL_PREFIX / (self.wavelength * 1e-9) ** 4 / np.expm1(EkT)

        for count in range(10):
            nmfilter = nm2eV / (Eguess - Vthlocal * kTfilter)  # MD [652., 930.]
            EQEfilter = self.corrEQE.copy()

            for i, lam in enumerate(self.wavelength):
                EQEfilter[i, :] *= lam < nmfilter  # zero EQE about nmfilter

            DBintegral = blackbody * EQEfilter
            Jdb = trapezoid(DBintegral, x=(self.wavelength * 1e-9), axis=0)
            Egnew = Egvect(TC, Jdb)
            if dbug:
                print(Egnew, max((Egnew - Eguess) / Egnew))
            if np.amax((Egnew - Eguess) / Egnew) < 1e-6:
                break
            else:
                Eguess = Egnew
            self.Egs = Egnew

        return Jdb, Egnew

    def Jint(self, enforce_all_combinations: bool = False) -> np.ndarray:
        """
        Integrates over spectrum or spectra to calculate the short-circuit current density (Jsc).

        J = spectra * lambda * EQE(lambda)
        Jsc = int(Pspec * EQE(lambda) * lambda) in [mA/cm2]

        Args:
            enforce_all_combinations (bool): If True, integrates all combinations of EQE and spectra.
                                             If False, integrates each EQE with its corresponding spectrum.

        Returns:
            np.ndarray: Integrated current density values.
        """

        if self.spectra is None:
            raise ValueError("Load spectral information first.")

        # Check if we have the same number of EQE curves as spectra
        if enforce_all_combinations or self.eqe.shape[1] != self.spectra.shape[1]:
            # Outer product for all combinations
            integrand = np.einsum("ni,nj->nij", (self.corrEQE / convert.wavelength_to_photonenergy(self.wavelength) * 1e-1), self.spectra)
            jsc = trapezoid(y=integrand, x=self.wavelength.flatten(), axis=0)  # Integrate along time axis
        else:

            # Pairwise integration: only integrate each EQE with its corresponding spectrum
            integrand = self.corrEQE * self.spectra / convert.wavelength_to_photonenergy(self.wavelength) * 1e-1
            jsc = trapezoid(y=integrand, x=self.wavelength.flatten(), axis=0)  # Integrate along wavelength axis

        return jsc

    def plot(
        self,
        Pspec: Union[str, np.ndarray] = "global",
        ispec: int = 0,
        specname: Union[str, None] = None,
        xspec: np.ndarray = wvl,
        size: str = "x-large",
        fig: Union[plt.Figure, None] = None,
        ax: Union[plt.Axes, None] = None,
    ) -> Tuple[plt.Axes, plt.Axes]:
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

    def plot_sr(self) -> Tuple[plt.Figure, plt.Axes]:
        # plot EQE on top of a spectrum

        sr = self.eqe * 1 / convert.wavelength_to_photonenergy(self.wavelength)
        fig, ax = plt.subplots()
        ax.plot(self.wavelength, sr)
        return fig, ax


class EQET(EQE):

    def __init__(self, wavelength: np.ndarray, eqe: np.ndarray, temperature: np.ndarray, name: str = "EQE", sjuncs: Union[List[str], None] = None):

        # ensure numpy
        temperature = np.atleast_1d(temperature) if not isinstance(temperature, np.ndarray) else temperature

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

        pass # pass for now

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
    #             print('Jsc = ',Jscs[0], ' mA/cm2')

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

        # interp_func = interp1d(self.temperature, self.eqe, axis=1, kind="linear", bounds_error=True)#, fill_value=0)
        interp_func = interp1d(self.temperature, self.eqe, axis=1, kind="linear", bounds_error=False, fill_value="extrapolate")  # , fill_value=0)
        eqe_new = interp_func(temperature)

        # interpolator = RegularGridInterpolator((self.wavelength.flatten(), self.temperature), self.eqe, method="linear", bounds_error=False, fill_value=None) # None = extrapolate
        # points = np.column_stack((self.wavelength.flatten(),np.full_like(self.wavelength.flatten(), temperature)))
        # eqe_new = interpolator(points)

        # clip negative
        eqe_new = np.clip(eqe_new, 0, self.eqe.max())

        # self.add_eqe(self.wavelength, eqe_new, temperature)
        return EQET(self.wavelength, eqe_new, temperature)

    def get_eqe_at_temperature(self, temperature: np.ndarray, method: str = "interpolate") -> EQET:
        if method not in self.method_dict:
            raise ValueError(f"Method '{method}' is not implemented. Available methods: {list(self.method_dict.keys())}")
        return self.method_dict[method](temperature)

    def get_current_for_temperature(self, target_temperature: Union[float, np.ndarray], degrees: Union[int, List[int]] = 5) -> np.ndarray:
        """
        Calculate the current density J(T) for each target temperature by first applying a polynomial fit
        to EQE(T,λ) as a function of temperature at each wavelength λλ,
        and then integrating over λ to obtain J(T)

        Args:
            target_temperature (Union[float,np.ndarray[float]]): target tempreature for J(T)
            degrees (Union[int, List[int]], optional): Degree of the polynomial fit. Defaults to 5. If a list is provided it will try all polynomials and chose the best fitting according BIC criterion.

        Returns:
            np.ndarray[float]: Array with J(T)
        """

        target_temperature = np.asarray(target_temperature)
        # check if spectra is available
        if self.spectra is None:
            raise ValueError("Load spectral information first.")

        # we only use the trace of the matrix, so spectra and temperature should have same dimension in one direction
        if not self.spectra.shape[1] == target_temperature.shape[0]:
            raise ValueError("Spectral and temperature array not the same length.")

        # calculate the current density for each spectrum and EQE(T)
        current_density = self.Jint(enforce_all_combinations=True)

        if isinstance(degrees, int):
            poly_degree = degrees
            best_fit_coeffs = np.array([np.polyfit(self.temperature, current_density[:, i], poly_degree) for i in range(current_density.shape[1])])

        else:
            # Try all degrees
            n = len(self.temperature)
            best_fit_coeffs = []

            # Loop over currents for each timestep
            for i in range(current_density.shape[1]):
                y = current_density[:, i]
                best_bic = np.inf
                best_coeff = None

                # Loop over all polynomial degrees and fit data
                for degree in degrees:
                    coeffs, residuals, rank, _, _ = np.polyfit(self.temperature, y, degree, full=True)
                    y_fit = np.polyval(coeffs, self.temperature)

                    if rank < degree + 1:
                        bic = np.inf
                    else:
                        # add offest to prevent log(0)
                        rss = residuals + 1e-12
                        # Calculate Bayesian information criterion (BIC)
                        k = degree + 1
                        bic = n * np.log(rss / n) + k * np.log(n)

                    # chose best fit based on BIC
                    if bic < best_bic:
                        best_bic = bic
                        best_coeff = coeffs

                best_fit_coeffs.append(best_coeff)

        # Calculate current_fits using the best-fit coefficients for each target temperature
        current_fits = np.array([np.polyval(best_fit_coeffs[i], target_temperature[i]) for i in range(len(target_temperature))])

        # exmaple fit for worst goodness of fit (gof)
        # gof = np.array([np.sum((np.polyval(fit_coeff, self.temperature) - current_density[:, i]) ** 2) for i, fit_coeff in enumerate(best_fit_coeffs)])

        # mse_max = np.argmax(gof)
        # plt.plot(self.temperature, current_density[:, mse_max], "o")
        # plt.plot(self.temperature, np.polyval(best_fit_coeffs[mse_max], self.temperature), "--")
        # plt.plot(-28, np.polyval(best_fit_coeffs[mse_max], -28), "*")
        # plt.title(f"Order = {len(best_fit_coeffs[mse_max])-1}")

        return current_fits

    def get_sr(self) -> np.ndarray:
        """
        Return spectral response (SR)

        Returns:
            np.ndarray: spectral response data
        """
        sr = self.eqe.T / convert.photonenergy_to_wavelength(self.wavelength.flatten())
        return sr.T

    def add_eqe(self, wavelength_add: np.ndarray, eqe_add: np.ndarray, temperature_add: Union[float, np.ndarray], sjuncs: Union[str, None] = None) -> None:
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

    def plot(self, fig: Union[plt.Figure, None] = None, ax: Union[plt.Axes, None] = None) -> Tuple[plt.Axes, plt.Axes]:
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
        fig.colorbar(sm, ax=ax, orientation="horizontal", location="top", pad=0.01, label="Temperature (℃)")

        # set labels
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("EQE")

        rax = ax.twinx()  # right axis
        return ax, rax

    def plot_sr(self, fig: Union[plt.Figure, None] = None, ax: Union[plt.Axes, None] = None) -> None:
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
        fig.colorbar(sm, ax=ax, orientation="horizontal", location="top", pad=0.01, label="Temperature (℃)")

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


def _polynomial_model(order: int) -> Tuple[Callable, List[float]]:
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


def _spline3(temperature: np.ndarray, *params: float) -> np.ndarray:
    temperature = np.asarray(temperature)
    x_vals = np.linspace(min(temperature), max(temperature), len(params))
    spline = UnivariateSpline(x_vals, params, k=3, s=0, ext=0)  # 0 for extrapolation
    return spline(temperature)


def _spline_model(spline_order: int) -> Tuple[Callable, List[float]]:
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
    PIECEWISE_LINEAR = ("piecewise_linear", _piecewise_linear_model, [120.0, 1.0, 0.0, 1.0, 0.0])  # t_split, p0, p1
    POLY2 = ("2nd_order_polynomial", _poly2, [1.0, 0.0, 0.0])
    POLY3 = ("3rd_order_polynomial", _poly3, [1.0, 0.0, 0.0, 0.0])
    POLY4 = ("4th_order_polynomial", _poly4, [1.0, 0.0, 0.0, 0.0, 0.0])
    POLY5 = ("5th_order_polynomial", _poly5, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # SPLINE3 = ("cubic_spline", _spline3, [1.0] * 5)

    def __init__(self, name: str, function: Callable, initial_guess: List[float]):
        self._function = function
        self._initial_guess = initial_guess

    @property
    def function(self) -> Callable:
        return self._function

    @property
    def initial_guess(self) -> List[float]:
        return self._initial_guess


def _model_residuals(params: Tuple[float, ...], func: Callable, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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


def _calc_mic(temperature: np.ndarray, predicted: np.ndarray, actual: np.ndarray, params: Tuple[float, ...]) -> Tuple[float, float]:
    """
    Calcualtes the model information criterion (mic). Can be either Akaike, Bayesian or penalized versions of these.

    Args:
        temperature (np.ndarray): temperature / x-values
        predicted (np.ndarray): Predicted data from the model
        actual (np.ndarray): Actual data for comparison
        params (Tuple): Parameters for the fit model

    Returns:
        Tuple[float, float]: Returns a tuple of MIC and mean squared error (MSE)
    """
    mse = np.mean((predicted - actual) ** 2)
    # Model information cirterion
    # mic = len(temperature) * np.log(mse + np.finfo(float).eps) + 2 * len(params)  # Akaike Information Criterion
    # mic = len(temperature) * np.log(mse + np.finfo(float).eps) + len(params) * np.log(len(temperature))  # Bayesian Information Criterion
    param_penalty = 1
    # mic = len(temperature) * (np.log(mse + np.finfo(float).eps))**1 + param_penalty * len(params)  # Akaike Information Criterion wiht additional parameter penalty
    mic = len(temperature) * np.log(mse + np.finfo(float).eps) + param_penalty * len(params) * np.log(len(temperature))  # Bayesian Information Criterion wiht additional parameter penalty
    return mic, mse


class TemperatureModel:
    """
    Temperature model for bandgap and sigma
    """

    def __init__(self, model_type: ModelType, params: List[float], Tref: float = 25):
        """
        Initializes the model with a specified type and parameters.

        Args:
            model_type (ModelType): Type of temperature dependence model.
            params (List[float]): Parameters for the chosen model.
            Tref (float): Reference temperature.
        """
        if model_type not in ModelType:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        self.model_type = model_type
        self.params = params
        self.Tref = Tref

    def apply(self, temperature: Union[float, np.ndarray], ref_value: float) -> Union[float, np.ndarray]:
        """
        Applies the specified model to temperature and scales it with a reference value.

        Args:
            temperature (Union[float, pd.Series]): Target temperature.
            ref_value (float): Reference value at a baseline temperature (e.g., 25°C).

        Returns:
            Union[float, pd.Series]: Calculated value at target temperature.
        """
        model_func = self.model_type.function
        temperature = np.array(temperature)
        return model_func(temperature, *self.params) * ref_value

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, plot: Union[str, None] = None, model_types: Union[ModelType, List[ModelType], None] = None, Tref: float = 25) -> "TemperatureModel":
        """
        Fits multiple models to the data and returns an instance of the best-fitting model, parameters, and MSE.

        Args:
            eqe: An instance containing temperature and bandgap data.

        Returns:
            Tuple[BandgapTemperatureModel, List[float], float]: Best-fitting model instance, parameters, and MSE.
        """
        best_mic = float("inf")
        best_params = []
        best_model_type = None

        if plot is not None:
            fig, ax = plt.subplots(1, 1, layout="constrained")
            ax.plot(x, y, "*", label="data")
            ax.set_xlabel("Temperature (℃)")
            ax.set_ylabel("Bandgap/sigma (eV)")

        if model_types is None:
            model_types = ModelType
        else:
            # ensure list
            model_types = model_types if isinstance(model_types, list) else [model_types]

        for model_type in model_types:
            func = model_type.function
            initial_guess = model_type.initial_guess

            try:
                result = least_squares(_model_residuals, initial_guess, args=(func, x, y), loss="soft_l1")
                params = result.x
                predicted = func(x, *params)
                mic, mse = _calc_mic(x, predicted, y, params)

                if mic < best_mic:
                    best_mic = mic
                    best_params = params
                    best_model_type = model_type

                if plot == "all":
                    t_space = np.linspace(x.min(), x.max())
                    pred = func(t_space, *params)
                    ax.plot(t_space, pred, "--", label=f"{model_type.name} MIC={mic:.2e} MSE={mse:.2e}")

            except RuntimeError:
                continue

        if best_model_type is None:
            raise ValueError("No model could be fit to the data.")

        if plot == "best":
            func = best_model_type.function
            t_space = np.linspace(x.min(), x.max())
            pred = func(t_space, *best_params)
            ax.plot(t_space, pred, "--", label=best_model_type.name)

        if plot is not None:
            ax.legend(fontsize=8)

        ref_value = best_model_type.function(Tref, *best_params)

        # normalize params to ref value
        if best_model_type is ModelType.PIECEWISE_LINEAR:
            normalized_params = best_params
            normalized_params[1:] /= ref_value
        else:
            normalized_params = best_params / ref_value

        temperature_model = cls(best_model_type, normalized_params)
        return temperature_model
