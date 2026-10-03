"""
This is the PVcircuit Package.
pvcircuit.Junction()
properties and methods for each junction
"""

from __future__ import annotations

import copy
import math  # simple math
import os
import warnings
from datetime import datetime
from functools import lru_cache
from time import time

import numpy as np  # arrays
from loguru import logger
from parse import parse
from scipy.integrate import quad  # numerical integration for non-Gaussian band tails
from scipy.optimize import brentq, elementwise  # root finders (scalar and vectorized)

from pvcircuit.conversions import DB_PREFIX, HC_E, K_Q, TK, Vth

# constants

# Junction defaults
Eg_DEFAULT = 1.1  # [eV]
SIGMA_DEFAULT = 0  # [eV]
TC_REF = 25.0  # [C]
AREA_DEFAULT = 1.0  # [cm^2] note: if A=1, then I->J
BETA_DEFAULT = 15.0  # unitless
# [A/cm^2] reference current density that makes J0ratio dimensionless.
# The value is 1 mA/cm^2, matching the historical PVcircuit/Igor convention.
J0_REFERENCE = 1e-3

# numerical calculation parameters
VLIM_REVERSE = 10.0
VLIM_FORWARD = 3.0
EPSREL = 1e-15
MAXITER = 1000
SCALAR_SOLVE_CUTOFF = 100
# Voltage tolerance for scalar Brent solves and 3T Brent fallbacks.
XTOL_SOLVE = 1e-11
# Newton junction solves (_vjunction_rows, _vjunction_scalar): voltage
# tolerance, and the number of Newton steps after which a point is finished by
# bisection alone.
XTOL_ROWS = 1e-13
NEWTON_ROWS = 50

# Relative tolerance of the final refinement in _maximize_rows. The power
# maximum is flat, so the power is accurate to about the square of this.
XRTOL_MPP = 3e-6

# Solver behind every junction solve and the device methods built on them.
#   "fast"   - safeguarded Newton (plain floats for a few points, arrays for
#              many), root finding in Multi2T.I2T, all timesteps at once in
#              EY.Meteo.run_ey
#   "legacy" - the Brent root finders, the stepping Multi2T.I2T and the
#              timestep-by-timestep run_ey used before
# Change it at run time: pvcircuit.junction.SOLVER = "legacy"
SOLVER = "fast"
SOLVERS = ("fast", "legacy")


def _legacy_solver() -> bool:
    """True if SOLVER selects the legacy algorithms; an unknown setting raises."""
    if SOLVER not in SOLVERS:
        raise ValueError(f"pvcircuit.junction.SOLVER must be one of {SOLVERS}, not {SOLVER!r}")
    return SOLVER == "legacy"


# repository root (parent of the pvcircuit package directory); pvc_output is created here
GITpath = os.path.dirname(os.path.dirname(__file__))


@lru_cache(maxsize=100)
def Jdb(TC: float, Eg: float, sigma: float = 0, theta: float = 2.0):
    """[A/cm^2] detailed-balance reverse saturation current density.

    Two physical models, selected by ``theta``:

    - ``theta = 2.0`` (default, Mattheis-Rau-Werner): ``sigma`` is the
      standard deviation [eV] of a Gaussian distribution of local bandgaps.
      Uses the historical pvcircuit closed-form expression. This is the
      backward-compatible path; existing call sites and serialized devices
      are unaffected.

    - ``theta != 2.0`` (generalized Urbach, Katahara et al. 2014 JAP 116,
      173504): ``sigma`` is reinterpreted as the energy scale gamma_U [eV]
      of an exponential-family sub-bandgap tail

          a(E) = exp(-((Eg - E) / sigma) ** theta)   for  E < Eg
          a(E) = 1                                     for  E >= Eg

      Integrated numerically against the Boltzmann blackbody tail.

      Suggested theta values (Katahara 2014):
          theta = 1     : true Urbach (GaAs ~ 9 meV, perovskite, a-Si)
          theta = 5/4   : screened Thomas-Fermi (CZTS-like)
          theta = 3/2   : Franz-Keldysh (CIGS)
          theta = 2     : Werner-Rau (uses the closed form above)

    For ``sigma == 0`` both models reduce to the Shockley-Queisser step
    result independent of ``theta``, so all existing zero-sigma callers and
    baseline tests are unchanged.

    A UserWarning is emitted in the numerical path when ``sigma > 3 * Vth``,
    because the generalized-Urbach absorptivity model becomes physically
    suspect when the tail energy exceeds a few thermal voltages.
    """

    Vthlocal = Vth(TC)
    TKlocal = TK(TC)
    EgkT = Eg / Vthlocal
    sq_bracket = EgkT * EgkT + 2.0 * EgkT + 2.0

    # Detailed balance step (no tail). Both theta branches collapse here.
    if sigma == 0:
        return DB_PREFIX * TKlocal**3.0 * sq_bracket * np.exp(-EgkT)

    if theta == 2.0:
        # Mattheis-Rau-Werner Gaussian bandgap-fluctuation closed form.
        # Historical pvcircuit expression, preserved verbatim.
        return DB_PREFIX * TKlocal**3.0 * (sq_bracket - 2 * sigma**2 * Eg / Vthlocal**3 - sigma**2 / Vthlocal**2 + sigma**4 / Vthlocal**4) * np.exp(-EgkT + sigma**2 / (2 * Vthlocal**2))

    # Generalized-Urbach numerical integration.
    if sigma > 3.0 * Vthlocal:
        warnings.warn(
            f"Jdb: sigma={sigma:.3g} eV exceeds 3*Vth ({3.0 * Vthlocal:.3g} eV) at TC={TC} C. Generalized-Urbach result may be unphysical.",
            UserWarning,
            stacklevel=2,
        )

    # Bracket = integral_0^inf a(x) * x^2 * exp(-x) dx with x = E/Vth.
    # Above-Eg piece (a=1, closed form): exp(-EgkT) * (EgkT^2 + 2 EgkT + 2).
    # Below-Eg piece (numerical): a(E) = exp(-((Eg-E)/sigma)^theta).
    def _integrand(x):
        u = EgkT - x  # = (Eg - E) / Vth, > 0 on the below-Eg interval
        ratio = u * Vthlocal / sigma  # = (Eg - E) / sigma
        return np.exp(-(ratio**theta)) * x * x * np.exp(-x)

    below_eg, _ = quad(_integrand, 0.0, EgkT, limit=200, epsrel=1e-10)
    return DB_PREFIX * TKlocal**3.0 * (np.exp(-EgkT) * sq_bracket + below_eg)


def timestamp(fmt="%y%m%d-%H%M%S", tm=None) -> str:
    # return a timestamp string with given format and epoch time
    if tm is None:
        tm = time()
    date_time = datetime.fromtimestamp(tm)
    return date_time.strftime(fmt)


def newoutpath(dname: str | None = None) -> str | None:
    # return a new output within pvc_output
    if os.path.exists(GITpath):
        pvcoutpath = os.path.join(GITpath, "pvc_output")
        if not os.path.exists(pvcoutpath):
            os.mkdir(pvcoutpath)

        if dname is None:
            dname = timestamp()
        else:
            dname += timestamp()

        newpath = os.path.join(pvcoutpath, dname)
        if not os.path.exists(newpath):
            os.mkdir(newpath)

        return newpath
    return None


def _saturation_current_from_ratio(Jdb_value, ideality_factor, ratio):
    """Return saturation current density [A/cm^2] from J0ratio.

    All current densities remain in A/cm^2. J0_REFERENCE appears only to
    make the fractional-power argument and J0ratio dimensionless. The direct
    path preserves historical/Igor results.
    """
    Jdb_value = np.asarray(Jdb_value, dtype=np.float64)
    ideality_factor = np.asarray(ideality_factor, dtype=np.float64)
    ratio = np.asarray(ratio, dtype=np.float64)
    reference_inverse = 1.0 / J0_REFERENCE
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        direct = (Jdb_value * reference_inverse) ** (1.0 / ideality_factor) * ratio / reference_inverse
        log_magnitude = math.log(J0_REFERENCE) + np.log(np.abs(ratio)) + np.log(Jdb_value / J0_REFERENCE) / ideality_factor
        stable = np.sign(ratio) * np.exp(log_magnitude)
        use_stable = ((direct == 0.0) & (stable != 0.0)) | (~np.isfinite(direct) & np.isfinite(stable))
        return np.where(use_stable, stable, direct)


def _ratio_from_saturation_current(Jdb_value, ideality_factor, saturation_current):
    """Return dimensionless J0ratio from saturation current [A/cm^2]."""
    Jdb_value = np.asarray(Jdb_value, dtype=np.float64)
    ideality_factor = np.asarray(ideality_factor, dtype=np.float64)
    saturation_current = np.asarray(saturation_current, dtype=np.float64)
    reference_inverse = 1.0 / J0_REFERENCE
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        direct = reference_inverse * saturation_current / (Jdb_value * reference_inverse) ** (1.0 / ideality_factor)
        log_magnitude = np.log(np.abs(saturation_current)) - math.log(J0_REFERENCE) - np.log(Jdb_value / J0_REFERENCE) / ideality_factor
        stable = np.sign(saturation_current) * np.exp(log_magnitude)
        use_stable = ((direct == 0.0) & (stable != 0.0)) | (~np.isfinite(direct) & np.isfinite(stable))
        return np.where(use_stable, stable, direct)


def _scaled_expm1(coefficient, exponent):
    """Evaluate coefficient * expm1(exponent) without intermediate overflow."""
    coefficient, exponent = np.broadcast_arrays(
        np.asarray(coefficient, dtype=np.float64),
        np.asarray(exponent, dtype=np.float64),
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        direct = coefficient * np.expm1(exponent)
        if np.isfinite(direct).all():
            # the log-domain form below is only ever selected where direct is not finite
            return np.asarray(direct)
        stable = np.sign(coefficient) * np.exp(np.log(np.abs(coefficient)) + exponent)
        stable = np.where(coefficient == 0.0, 0.0, stable)
        use_stable = (exponent > 0.0) & ~np.isfinite(direct) & np.isfinite(stable)
        return np.where(use_stable, stable, direct)


def _scaled_exp(coefficient, exponent):
    """Evaluate coefficient * exp(exponent) without intermediate range loss."""
    coefficient, exponent = np.broadcast_arrays(
        np.asarray(coefficient, dtype=np.float64),
        np.asarray(exponent, dtype=np.float64),
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        direct = coefficient * np.exp(exponent)
        if (np.isfinite(direct) & (direct != 0.0)).all():
            # the log-domain form below is only selected where direct is zero or not finite
            return np.asarray(direct)
        stable = np.sign(coefficient) * np.exp(np.log(np.abs(coefficient)) + exponent)
        stable = np.where(coefficient == 0.0, 0.0, stable)
        use_stable = ((direct == 0.0) & (stable != 0.0)) | (~np.isfinite(direct) & np.isfinite(stable))
        return np.where(use_stable, stable, direct)


def _scaled_expm1_scalar(coefficient, exponent):
    """Plain-float counterpart of _scaled_expm1 for scalar Brent residuals."""
    if coefficient == 0.0:
        return 0.0
    try:
        direct = coefficient * math.expm1(exponent)
    except OverflowError:
        direct = math.copysign(math.inf, coefficient)
    if math.isfinite(direct) or exponent <= 0.0:
        return direct
    log_magnitude = math.log(abs(coefficient)) + exponent
    if log_magnitude <= math.log(np.finfo(np.float64).max):
        return math.copysign(math.exp(log_magnitude), coefficient)
    return direct


# ----------------------------------------------------------------------
# Vectorized solver core
# ----------------------------------------------------------------------


def _recomb_current(V, st, plus_J0=False):
    """[A/cm^2] vectorized total recombination + shunt + RBB current density.

    Equivalent to Junction.Jmultidiodes(V) + Junction.JshuntRBB(V), evaluated
    from the parameter snapshot ``st`` (see Junction._solver_state). ``V`` is
    the junction-frame diode voltage; scalar or ndarray. With a per-row
    state (Junction._solver_state_rows) ``V`` holds one voltage per row.

    ``plus_J0=True`` returns the same current plus sum(J0): the diode terms
    are then J0 * exp(V / nVth) instead of J0 * expm1(V / nVth), without the
    cancellation that adding sum(J0) afterwards suffers in reverse bias.
    """
    V = np.asarray(V, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if st["J0"].size:
            exponent = V[..., None] / st["nVth"]
            Jled = np.sum(_scaled_exp(st["J0"], exponent) if plus_J0 else _scaled_expm1(st["J0"], exponent), axis=-1)
        else:
            Jled = np.zeros_like(V)
        out = Jled + V * st["Gsh"]

        method = st["rbb"].get("method")
        if method == "JFG":
            Vrb = st["rbb"]["Vrb"]
            mrb = st["rbb"]["mrb"]
            if mrb != 0.0:
                K = st["rbb"]["J0rb_effective"]
                out = out + np.where(V <= Vrb, _scaled_expm1(-K, -V / (st["Vth"] * mrb)), 0.0)
        elif method == "bishop":
            Vrb = st["rbb"]["Vrb"]
            av = st["rbb"]["avalanche"]
            mrb = st["rbb"]["mrb"]
            if Vrb != 0.0:
                mask = V <= 0.0
                base = np.where(mask, 1.0 + V / Vrb, 1.0)  # keep base positive off-branch
                out = out + np.where(mask, V * st["Gsh"] * av * base ** (-mrb), 0.0)
        elif method == "pvmismatch":
            raise NotImplementedError("RBB method 'pvmismatch' is documented but not implemented. Use RBB='JFG', RBB='bishop', or RBB=None.")
    return out


def _recomb_current_deriv(V, st):
    """[A/cm^2/V] analytic derivative of _recomb_current w.r.t. V.

    Used as the Jacobian of the Newton solvers; smooth everywhere except the
    RBB branch boundaries (handled by damping in the callers).
    """
    V = np.asarray(V, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if st["J0"].size:
            denominator = st["nVth"]
            g = np.sum(_scaled_exp(st["J0"] / denominator, V[..., None] / denominator), axis=-1)
        else:
            g = np.zeros_like(V)
        g = g + st["Gsh"]

        method = st["rbb"].get("method")
        if method == "JFG":
            Vrb = st["rbb"]["Vrb"]
            mrb = st["rbb"]["mrb"]
            if mrb != 0.0:
                K = st["rbb"]["J0rb_effective"]
                denominator = st["Vth"] * mrb
                g = g + np.where(V <= Vrb, _scaled_exp(K / denominator, -V / denominator), 0.0)
        elif method == "bishop":
            Vrb = st["rbb"]["Vrb"]
            av = st["rbb"]["avalanche"]
            mrb = st["rbb"]["mrb"]
            if Vrb != 0.0:
                mask = V <= 0.0
                base = np.where(mask, 1.0 + V / Vrb, 1.0)
                g = g + np.where(mask, st["Gsh"] * av * base ** (-mrb) - V * st["Gsh"] * av * mrb * base ** (-mrb - 1.0) / Vrb, 0.0)
    return g


def _recomb_current_scalar(V, st):
    """Scalar twin of _recomb_current using plain-python math.

    brentq evaluates the residual ~15 times per solve; numpy scalar-op
    overhead dominates there, so this pure-float version (~10x faster per
    call) is used by the small-N loops in _vdiode_arr/_vmid_arr. Uses the
    same expm1 formulation as the vectorized version so both paths converge
    to identical roots.
    """
    out = 0.0
    for J0f, binv in zip(st["J0f"], st["ninv"]):
        x = V * binv
        out += _scaled_expm1_scalar(J0f, x)
    out += V * st["Gsh"]

    method = st["rbb"].get("method")
    if method == "JFG":
        Vrb = st["rbb"]["Vrb"]
        mrb = st["rbb"]["mrb"]
        if V <= Vrb and mrb != 0.0:
            K = st["rbb"]["J0rb_effective"]
            x = -V / (st["Vth"] * mrb)
            out += _scaled_expm1_scalar(-K, x)
    elif method == "bishop":
        Vrb = st["rbb"]["Vrb"]
        if V <= 0.0 and Vrb != 0.0:
            out += V * st["Gsh"] * st["rbb"]["avalanche"] * (1.0 + V / Vrb) ** (-st["rbb"]["mrb"])
    elif method == "pvmismatch":
        raise NotImplementedError("RBB method 'pvmismatch' is documented but not implemented. Use RBB='JFG', RBB='bishop', or RBB=None.")
    return out


def _exp_scalar(x):
    """math.exp that returns inf instead of raising on overflow."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def _recomb_newton_scalar(V, st):
    """Plain-float (S, g) for the scalar Newton solve of a single-condition state.

    S = J(V) + sum(J0) [A/cm^2] and g = dJ/dV [A/cm^2/V], the same terms as
    _recomb_current(V, st, plus_J0=True) and _recomb_current_deriv(V, st).
    (nan, nan) where the Bishop breakdown term is undefined.
    """
    Gsh = st["Gsh"]
    S = V * Gsh
    g = Gsh
    for J0f, binv in zip(st["J0f"], st["ninv"]):
        if J0f != 0.0:
            e = J0f * _exp_scalar(V * binv)
            S += e
            g += e * binv

    method = st["rbb"].get("method")
    if method == "JFG":
        mrb = st["rbb"]["mrb"]
        if V <= st["rbb"]["Vrb"] and mrb != 0.0:
            K = st["rbb"]["J0rb_effective"]
            if K != 0.0:
                binv = 1.0 / (st["Vth"] * mrb)
                S += _scaled_expm1_scalar(-K, -V * binv)
                coefficient = K * binv
                term = coefficient * _exp_scalar(-V * binv)
                if not math.isfinite(term) and coefficient > 0.0:
                    # the exponential alone overflows before the product: log domain, like S
                    term = _exp_scalar(math.log(coefficient) - V * binv)
                g += term
    elif method == "bishop":
        Vrb = st["rbb"]["Vrb"]
        if V <= 0.0 and Vrb != 0.0:
            base = 1.0 + V / Vrb
            if base <= 0.0:
                return math.nan, math.nan
            mrb = st["rbb"]["mrb"]
            shunt = Gsh * st["rbb"]["avalanche"] * base ** (-mrb)
            S += V * shunt
            g += shunt - V * shunt * mrb / (base * Vrb)
    elif method == "pvmismatch":
        raise NotImplementedError("RBB method 'pvmismatch' is documented but not implemented. Use RBB='JFG', RBB='bishop', or RBB=None.")
    return S, g


def _vjunction_scalar(Jtot, st, Gx=0.0, Vref=0.0):
    """Plain-float twin of _vjunction_rows for one point of a single-condition state.

    Solves J(V) + Gx * (V - Vref) = Jtot with the same safeguarded Newton
    iteration; for a handful of points this is faster than array operations.
    Returns nan where [-VLIM_REVERSE, VLIM_FORWARD] holds no root.
    """
    if not (math.isfinite(Jtot) and math.isfinite(Vref)):
        return math.nan
    lo = -VLIM_REVERSE
    hi = VLIM_FORWARD
    Tj = Jtot + st["J0sum"]
    if not (_recomb_newton_scalar(lo, st)[0] - Tj) + Gx * (lo - Vref) <= 0.0:
        return math.nan
    if not (_recomb_newton_scalar(hi, st)[0] - Tj) + Gx * (hi - Vref) >= 0.0:
        return math.nan

    # start value, as in _vjunction_rows
    Jeff = Jtot + Gx * Vref
    Geff = st["Gsh"] + Gx
    V = math.inf
    for J0f, binv in zip(st["J0f"], st["ninv"]):
        if J0f > 0.0:
            x = Jeff / J0f
            if x <= -1.0:
                V = math.nan
                break
            Vdiode = math.log1p(x) / binv
            if Vdiode < V:
                V = Vdiode
    if Geff > 0.0:
        Vlinear = Jeff / Geff
        if Jeff <= 0.0 or Vlinear < V:
            V = Vlinear
    V = min(max(V, lo), hi) if math.isfinite(V) else 0.0

    Tt = Tj + Gx * Vref
    for iteration in range(MAXITER):
        S, g = _recomb_newton_scalar(V, st)
        f = (S - Tj) + Gx * (V - Vref)
        if f > 0.0:
            hi = V
        elif f < 0.0:
            lo = V
        elif f == 0.0:
            return V
        else:  # undefined residual
            return math.nan
        gt = g + Gx
        Vn = math.nan
        if gt != 0.0 and math.isfinite(gt):  # an infinite derivative gives a zero step: bisect
            step = f / gt
            St = S + Gx * V
            if St > 0.0 and Tt > 0.0:
                ratio = St / Tt
                if ratio > 0.0:
                    logstep = math.log(ratio) * (St / gt)
                    if math.isfinite(logstep):
                        step = logstep
            Vn = V - step
            if abs(step) <= XTOL_ROWS:
                return Vn
        if not (lo <= Vn <= hi) or iteration >= NEWTON_ROWS:
            Vn = 0.5 * (lo + hi)
        if hi - lo <= XTOL_ROWS:
            return Vn
        V = Vn
    return V


def _jem_arr(Vmid, Jphoto, st):
    """[A/cm^2] vectorized Junction.Jem: PL (gamma*Jphoto) + EL for Vmid > 0."""
    Vmid = np.asarray(Vmid, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        el = np.where(Vmid > 0.0, _scaled_expm1(st["Jdb"], Vmid / st["Vth"]), 0.0)
    return st["gamma"] * Jphoto + el


def _jem_deriv(Vmid, st):
    """[A/cm^2/V] derivative of the EL part of _jem_arr w.r.t. Vmid."""
    Vmid = np.asarray(Vmid, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.where(Vmid > 0.0, _scaled_exp(st["Jdb"] / st["Vth"], Vmid / st["Vth"]), 0.0)


# ----------------------------------------------------------------------
# Per-row solver core (many operating conditions at once)
# ----------------------------------------------------------------------


def _jdb_rows(TC, Eg, sigma, theta=2.0):
    """[A/cm^2] Jdb for row arrays of (TC, Eg, sigma): the vectorized twin of Jdb.

    The closed forms (sigma == 0, or theta == 2) are evaluated for all rows
    at once. Generalized-Urbach rows (theta != 2 with sigma != 0) call the
    scalar Jdb row by row because of its numerical integral.
    """
    TC, Eg, sigma = (np.array(a, dtype=np.float64) for a in np.broadcast_arrays(TC, Eg, sigma))
    Vthlocal = Vth(TC)
    TKlocal = TK(TC)
    EgkT = Eg / Vthlocal
    sq_bracket = EgkT * EgkT + 2.0 * EgkT + 2.0
    out = DB_PREFIX * TKlocal**3.0 * sq_bracket * np.exp(-EgkT)
    tail = sigma != 0.0
    if tail.any():
        if theta == 2.0:
            s, v = sigma[tail], Vthlocal[tail]
            out[tail] = DB_PREFIX * TKlocal[tail] ** 3.0 * (sq_bracket[tail] - 2 * s**2 * Eg[tail] / v**3 - s**2 / v**2 + s**4 / v**4) * np.exp(-EgkT[tail] + s**2 / (2 * v**2))
        else:
            out[tail] = [Jdb(float(t), float(e), float(s), theta=float(theta)) for t, e, s in zip(TC[tail], Eg[tail], sigma[tail])]
    return out


def _state_take(st, idx):
    """Per-point copy of a per-row solver state: point k uses row idx[k] of ``st``.

    ``idx`` is an integer index array or a boolean mask over the rows.
    """
    out = dict(st)
    for key in ("nVth", "J0", "Vth", "Jdb", "notdiode"):
        out[key] = st[key][idx]
    if "J0rb_effective" in st["rbb"]:
        out["rbb"] = dict(st["rbb"], J0rb_effective=st["rbb"]["J0rb_effective"][idx])
    return out


def _vjunction_rows(Jtot, st, Gx=0.0, Vref=0.0):
    """Solve J(V) + Gx * (V - Vref) = Jtot for the junction voltage of every element.

    J(V) is _recomb_current(V, st). ``st`` is a per-row state
    (Junction._solver_state_rows; element k uses row k) or a single-condition
    state (Junction._solver_state; all elements share it). Gx = 0 is the
    current-driven junction (_vdiode_rows); Gx = 1 / Rser with Vref = Vtot and
    Jtot = Jphoto is the voltage-driven junction behind a series resistance
    (_vmid_rows).

    Works on S(V) = J(V) + sum(J0), which stays well conditioned in reverse
    bias where J itself saturates at -sum(J0). Safeguarded Newton inside the
    bracket [-VLIM_REVERSE, VLIM_FORWARD]: the step is taken on the logarithm
    of the current, which is exact for a single exponential; it falls back to
    a plain Newton step and then to bisection of the maintained bracket.
    Returns np.nan where the bracket holds no root, the same contract as
    Junction._vdiode_arr.
    """
    Jtot = np.asarray(Jtot, dtype=np.float64)
    npts = Jtot.size
    per_row = st["J0"].ndim == 2
    resistive = Gx != 0.0
    Vref = np.broadcast_to(np.asarray(Vref, dtype=np.float64), Jtot.shape)
    V = np.full(npts, np.nan)
    lo = np.full(npts, -VLIM_REVERSE)
    hi = np.full(npts, VLIM_FORWARD)
    T = Jtot + st["J0"].sum(axis=-1)
    with np.errstate(invalid="ignore", over="ignore"):
        flo = _recomb_current(lo, st, plus_J0=True) - T
        fhi = _recomb_current(hi, st, plus_J0=True) - T
        if resistive:
            flo = flo + Gx * (lo - Vref)
            fhi = fhi + Gx * (hi - Vref)
        bracketed = np.isfinite(T) & np.isfinite(Vref) & (flo <= 0.0) & (fhi >= 0.0)

    # start value: the smallest single-diode voltage (an upper bound of the root
    # for a positive current), limited by the linear branch (shunt, series
    # resistance); the linear branch alone in reverse
    Jeff = Jtot + Gx * Vref if resistive else Jtot
    Geff = st["Gsh"] + Gx
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if st["J0"].size:
            Vdiodes = np.log1p(Jeff[:, None] / st["J0"]) * st["nVth"]
            V0 = np.min(np.where(st["J0"] > 0.0, Vdiodes, np.inf), axis=-1)
        else:
            V0 = np.full(npts, np.inf)
        if Geff > 0.0:
            V0 = np.where(Jeff > 0.0, np.minimum(V0, Jeff / Geff), Jeff / Geff)
    V0 = np.where(np.isfinite(V0), np.clip(V0, -VLIM_REVERSE, VLIM_FORWARD), 0.0)

    act = np.flatnonzero(bracketed)
    sta = _state_take(st, act) if per_row else st
    Va, la, ha, Ta, Vra = V0[act], lo[act], hi[act], T[act], Vref[act]
    for iteration in range(MAXITER):
        if act.size == 0:
            break
        S = _recomb_current(Va, sta, plus_J0=True)
        g = _recomb_current_deriv(Va, sta)
        f = S - Ta
        if resistive:
            # total current through diode, shunt and series resistance against its target
            f = f + Gx * (Va - Vra)
            S = S + Gx * Va
            g = g + Gx
            Tt = Ta + Gx * Vra
        else:
            Tt = Ta
        ha = np.where(f > 0.0, Va, ha)
        la = np.where(f < 0.0, Va, la)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            logstep = np.log(S / Tt) * (S / g)
            step = np.where((S > 0.0) & (Tt > 0.0) & np.isfinite(logstep), logstep, f / g)
            # an overflowing derivative makes the step 0 although f is finite: that is
            # no convergence; the nan step fails both tests below and bisects
            step = np.where(np.isfinite(g), step, np.nan)
        Vn = Va - step
        # a step below the tolerance is converged even if rounding puts it on the
        # bracket edge; after NEWTON_ROWS steps only bisection is left, which ends
        small = np.abs(step) <= XTOL_ROWS
        newton = (Vn >= la) & (Vn <= ha) & (iteration < NEWTON_ROWS)
        Vn = np.where(small | newton, Vn, 0.5 * (la + ha))
        Vn = np.where(f == 0.0, Va, Vn)
        done = small | (f == 0.0) | (ha - la <= XTOL_ROWS)
        V[act] = Vn
        if done.any():
            keep = ~done
            act, Va, la, ha, Ta, Vra = act[keep], Vn[keep], la[keep], ha[keep], Ta[keep], Vra[keep]
            if per_row:
                sta = _state_take(sta, keep)
        else:
            Va = Vn
    return V


def _vdiode_rows(Jtot, st):
    """Solve _recomb_current(V, st) = Jtot: the current-driven junction, one voltage per element.

    Vectorized twin of Junction._vdiode_arr for a per-row or a single-condition
    state; np.nan where the bracket holds no root. See _vjunction_rows.
    """
    return _vjunction_rows(Jtot, st)


def _vmid_rows(Vtot, Jphoto, st, Rser):
    """Solve Vtot - V + (Jphoto - _recomb_current(V, st)) * Rser = 0, one voltage per element.

    Vectorized twin of Junction._vmid_arr for a per-row or a single-condition
    state: the junction voltage behind the series resistance ``Rser``
    [Ohm cm^2] at the terminal voltage ``Vtot``. np.nan where the bracket
    holds no root. See _vjunction_rows.
    """
    Vtot = np.asarray(Vtot, dtype=np.float64)
    Jphoto = np.broadcast_to(np.asarray(Jphoto, dtype=np.float64), Vtot.shape)
    if Rser == 0.0:
        with np.errstate(invalid="ignore"):
            inside = (Vtot >= -VLIM_REVERSE) & (Vtot <= VLIM_FORWARD) & np.isfinite(Jphoto)
        return np.where(inside, Vtot, np.nan)
    return _vjunction_rows(Jphoto, st, 1.0 / Rser, Vtot)


def _maximize_rows(power, lower, upper, index, pnts, rounds):
    """Maximum of power(u, index) for u between ``lower`` and ``upper``, for every element at once.

    ``power(u, index)`` returns the power of the elements ``index`` (integer
    array) at the abscissae ``u``; np.nan counts as no power. This is the
    search of Multi2T.MPP and of Tandem3T.CM / VM with all elements solved
    together: ``rounds`` scans of ``pnts`` points, each between the neighbours
    of the best point of the scan before. A bracketed scalar minimizer then
    refines the best point between its last neighbours, where those searches
    stop on their grid.

    Returns (u, P, complete): location and value of the maximum (np.nan
    without any solution), and whether every scan point before the best one
    had a solution. Points without one beyond the maximum are expected (a
    junction driven out of its voltage range); before it they are a solver
    failure that may hide the maximum.
    """
    count = np.arange(index.size)
    position = np.arange(pnts)[None, :]
    lo = np.array(np.broadcast_to(np.asarray(lower, dtype=np.float64), index.shape))
    hi = np.array(np.broadcast_to(np.asarray(upper, dtype=np.float64), index.shape))
    best = np.full(index.size, np.nan)
    Pbest = np.full(index.size, -np.inf)
    complete = np.ones(index.size, dtype=bool)
    for _ in range(rounds):
        grid = np.linspace(lo, hi, pnts, axis=1)
        P = power(grid.ravel(), np.repeat(index, pnts)).reshape(index.size, pnts)
        solved = np.isfinite(P)
        P = np.where(solved, P, -np.inf)
        nmax = np.argmax(P, axis=1)
        complete &= (solved | (position > nmax[:, None])).all(axis=1)
        higher = P[count, nmax] > Pbest
        best = np.where(higher, grid[count, nmax], best)
        Pbest = np.where(higher, P[count, nmax], Pbest)
        lo = grid[count, np.maximum(nmax - 1, 0)]
        hi = grid[count, np.minimum(nmax + 1, pnts - 1)]

    # refine inside the last pair of neighbours
    xm = grid[count, nmax]
    xl, xr = np.minimum(lo, hi), np.maximum(lo, hi)
    inside = (nmax > 0) & (nmax < pnts - 1) & (P[count, nmax] > 0.0) & (xl < xm) & (xm < xr)
    if inside.any():
        i = np.flatnonzero(inside)

        def negative_power(u, idx):
            value = power(u, idx.astype(np.intp))
            return np.where(np.isfinite(value), -value, 0.0)

        res = elementwise.find_minimum(negative_power, (xl[i], xm[i], xr[i]), args=(index[i].astype(np.float64),), tolerances=dict(xatol=0.0, xrtol=XRTOL_MPP, fatol=0.0, frtol=0.0))
        higher = -res.f_x > Pbest[i]
        best[i] = np.where(higher, res.x, best[i])
        Pbest[i] = np.where(higher, -res.f_x, Pbest[i])
    return best, np.where(np.isfinite(Pbest), Pbest, np.nan), complete


class Junction:
    """
    Class for PV junctions.
    """

    ATTR = ["Eg", "sigma", "TC", "Gsh", "Rser", "area", "lightarea", "totalarea", "Jext", "JLC", "beta", "gamma", "theta", "pn", "Jphoto", "TK", "Jdb", "RBB"]
    ARY_ATTR = ["n", "J0ratio", "J0"]

    # Diode arrays kept in sync via the custom __setattr__/set() pipeline.
    # Declared here so type checkers can subscript ``self.n[i]`` /
    # ``self.J0ratio[i]`` in ``__str__`` and other consumers.
    n: np.ndarray
    J0ratio: np.ndarray

    # Class-level default so legacy pickled Junctions (created before the
    # generalized-Urbach `theta` was added) still work. New instances
    # shadow this with ``self.theta = np.float64(theta)`` in __init__.
    theta: float = 2.0

    def __init__(
        self,
        name: str = "junc",
        Eg: float = Eg_DEFAULT,
        sigma: float = SIGMA_DEFAULT,
        TC: float = TC_REF,
        Gsh: float = 0.0,
        Rser: float = 0.0,
        area: float = AREA_DEFAULT,
        n: list[float] | None = None,
        J0ratio: list[float] | np.ndarray | None = None,
        J0ref: list[float] | np.ndarray | None = None,
        RBB: str | None = None,
        Jext: float = 0.04,
        JLC: float = 0.0,
        J0default: float = 10.0,
        pn: int = -1,
        beta: float = BETA_DEFAULT,
        gamma: float = 0.0,
        theta: float = 2.0,
    ):

        self.ui = None
        # self.debugout = widgets.Output()  # debug output
        self.RBB_dict = {}

        # user inputs
        self.name = name  # remember my name
        self.Eg = np.float64(Eg)  #: [eV] junction band gap
        self.sigma = np.float64(sigma)  #: [eV] junction band gap sigma (Urbach tail width)
        self.TC = np.float64(TC)  #: [\degC] junction temperature (use the ``TK`` property for Kelvin)
        self.Jext = np.float64(Jext)  #: [A/cm^2] external photocurrent density (printed as mA/cm^2)
        self.Gsh = np.float64(Gsh)  #: [S/cm^2] shunt conductance (= 1/Rsh, area-normalised)
        self.Rser = np.float64(Rser)  #: [\Omega*cm^2] series resistance (area-normalised, so ``Vdrop = Rser * J``)
        self.lightarea = np.float64(area)  #: [cm^2] illuminated junction area
        self.totalarea = np.float64(area)  #: [cm^2] total junction area including shaded regions
        # used for tandems only
        self.pn = int(pn)  #: polarity flag: +1 for p-on-n, -1 for n-on-p (sign convention)
        self.beta = np.float64(beta)  #: [unitless] luminescent coupling efficiency (top -> bottom radiative coupling)
        self.gamma = np.float64(gamma)  #: [unitless] photoluminescent coupling coefficient (Lan et al. PL parameter)
        self.theta = np.float64(theta)  #: [unitless] band-tail shape exponent (default 2.0 = Mattheis-Rau-Werner Gaussian; 1.0 = true Urbach; see Jdb())
        self.JLC = np.float64(JLC)  #: [A/cm^2] luminescent coupling current density injected from the previous junction (``beta * Jem`` of the neighbour)

        # multiple diodes
        # n=1 bulk, n=m SNS, and n=2/3 Auger mechanisms
        if n is None:
            n = [1.0, 2.0]
        ndiodes = len(n)
        self.n = np.array(n)  # diode ideality list e.g. [n0, n1]
        # 'is not None' rather than truthiness: numpy arrays raise on bool()
        if J0ref is not None and len(J0ref) > 0:  # input list of absolute J0
            if len(J0ref) == ndiodes:  # check length
                self._J0init(J0ref)  # calculate self.J0ratio from J0ref at current self.TC
            else:
                logger.warning("J0ref length {} does not match number of diodes {}; using default J0ratio", len(J0ref), ndiodes)
                self.J0ratio = np.full_like(n, J0default)  # default J0ratio
        elif J0ratio is not None and len(J0ratio) > 0:  # input list of relative J0 ratios
            if len(J0ratio) == ndiodes:  # check length
                self.J0ratio = np.array(J0ratio)  # diode J0/Jdb^(1/n) ratio list for T dependence
            else:
                logger.warning("J0ratio length {} does not match number of diodes {}; using default J0ratio", len(J0ratio), ndiodes)
                self.J0ratio = np.full_like(n, J0default)  # default J0ratio
        else:  # create J0ratio
            self.J0ratio = np.full_like(n, J0default)  # default J0ratio

        self.set(RBB=RBB)

    def copy(self) -> Junction:
        """
        create a copy of a Junction
        need deepcopy() to separate lists, dicts, etc but crashes
        """
        tmp = copy.copy(self)
        # manual since deepcopy does not work
        tmp.n = self.n.copy()
        tmp.J0ratio = self.J0ratio.copy()
        tmp.RBB_dict = self.RBB_dict.copy()
        return tmp

    def __str__(self):
        # attr_list = self.__dict__.keys()
        # attr_dict = self.__dict__.items()
        # print(attr_list)

        strout = self.name + ": <pvcircuit.junction.Junction class>"

        strout += f"\nEg = {self.Eg:.2f} eV, TC = {self.TC:.1f} C"

        strout += f"\nJext = {self.Jext * 1000.0:.1f} mA/cm^2, JLC = {self.JLC * 1000.0:.1f} mA/cm^2"

        strout += f"\nGsh = {self.Gsh:g} S/cm^2, Rser = {self.Rser:g} Ohm*cm^2"

        strout += f"\nlightA = {self.lightarea:g} cm^2, totalA = {self.totalarea:g} cm^2"

        strout += f"\npn = {self.pn:d}, beta = {self.beta:g}, gamma = {self.gamma:g}"
        # theta only shown when non-default so existing __str__ baselines stay byte-identical.
        if self.theta != 2.0:
            strout += f", theta = {self.theta:g}"

        strout += "\n {0:^5s} {1:^10s} {2:^10s}".format("n", "J0ratio", "J0(A/cm^2)")
        strout += "\n {0:^5s} {1:^10.0f} {2:^10.3e}".format("db", 1.0, self.Jdb)

        for i, _ in enumerate(self.n):
            strout += f"\n {self.n[i]:^5.2f} {self.J0ratio[i]:^10.2f} {self.J0[i]:^10.3e}"

        if self.RBB_dict["method"]:
            strout += " \nRBB_dict: " + str(self.RBB_dict)

        return strout

    def __repr__(self):
        return str(self)

    # """
    # def __setattr__(self, key, value):
    #     # causes problems
    #     super(Junction, self).__setattr__(key, value)
    #     self.set(key = value)
    # """

    # def update(self):
    #     # update Junction self.ui controls

    #     if self.ui:  # junction user interface has been created
    #         if self.RBB_dict:
    #             if self.RBB_dict["method"]:
    #                 RBB_keys = list(self.RBB_dict.keys())
    #             else:
    #                 RBB_keys = []

    #         cntrls = self.ui.children
    #         for cntrl in cntrls:
    #             desc = cntrl.trait_values().get("description", "nodesc")  # control description
    #             cval = cntrl.trait_values().get("value", "noval")  # control value
    #             if desc == "nodesc" or cval == "noval":
    #                 break
    #             elif desc.endswith("]") and desc.find("[") > 0:
    #                 key, ind = parse("{}[{:d}]", desc)
    #             else:
    #                 key = desc
    #                 ind = None

    #             if key in self.ATTR:  # Junction scalar controls to update
    #                 attrval = getattr(self, key)  # current value of attribute
    #                 if cval != attrval:
    #                     with self.debugout:
    #                         print("Jupdate: " + desc, attrval)
    #                     cntrl.value = attrval
    #             elif key in self.ARY_ATTR:  # Junction array controls to update
    #                 attrval = getattr(self, key)  # current value of attribute
    #                 if isinstance(ind, int):
    #                     if isinstance(attrval, np.ndarray):
    #                         if cval != attrval[ind]:
    #                             with self.debugout:
    #                                 print("Jupdate: " + desc, attrval[ind])
    #                             cntrl.value = attrval[ind]
    #             elif key in RBB_keys:
    #                 attrval = self.RBB_dict[key]
    #                 if cval != attrval:
    #                     with self.debugout:
    #                         print("Jupdate: " + desc, attrval)
    #                     cntrl.value = attrval

    def set(self, **kwargs):
        # controlled update of Junction attributes

        # with self.debugout:
        #     print("Jset(" + self.name + "): ", list(kwargs.keys()))

        for testkey, value in kwargs.items():
            if testkey.endswith("]") and testkey.find("[") > 0:
                parsed = parse("{}[{:d}]", testkey)  # set one element of array e.g. 'n[0]'
                if parsed is None:
                    raise ValueError(f"Could not parse array key {testkey!r}; expected form 'name[index]'")
                key, ind = parsed
            else:
                key = testkey
                ind = None

            if self.RBB_dict:
                if self.RBB_dict["method"]:
                    RBB_keys = list(self.RBB_dict.keys())
                else:
                    RBB_keys = []

            if key == "RBB" or key == "method":
                # this change requires redrawing self.ui
                if value == "JFG":  # RBB shortcut
                    self.__dict__["RBB_dict"] = {"method": "JFG", "mrb": 10.0, "J0rb": 0.5, "Vrb": 0.0}
                elif value == "bishop":
                    self.__dict__["RBB_dict"] = {"method": "bishop", "mrb": 3.28, "avalanche": 1.0, "Vrb": -5.5}
                else:
                    self.__dict__["RBB_dict"] = {"method": None}  # no RBB
                if self.ui:  # junction user interface has been created
                    # ui = self.controls()    # redraw junction controls
                    pass
            elif key in RBB_keys:  # RBB parameters
                self.RBB_dict[key] = np.float64(value)
            elif key == "area":  # area shortcut
                self.__dict__["lightarea"] = np.float64(value)
                self.__dict__["totalarea"] = np.float64(value)
            elif key == "name":  # strings
                self.__dict__[key] = str(value)
            elif key == "pn":  # integers
                self.__dict__[key] = int(value)
            elif key == "RBB_dict":
                self.__dict__[key] = value
            elif key in ["n", "J0ratio"]:  # diode parameters (array)
                if isinstance(ind, int) and np.isscalar(value):
                    attrval = getattr(self, key)  # current value of attribute
                    localarray = attrval.copy()
                    if isinstance(localarray, np.ndarray):
                        if ind < localarray.size:
                            # np.isscalar above guards at runtime; ty can't narrow it.
                            localarray[ind] = np.float64(value)  # ty: ignore[invalid-argument-type]
                            self.__dict__[key] = localarray
                            # with self.debugout:
                            #     print("scalar", key, ind, localarray)
                        else:
                            raise IndexError(f"invalid junction index. Set index is {ind + 1} but junction size is {localarray.size}")
                else:
                    # check if both, n and J0ratio, are set if they have the same size
                    if "n" in kwargs and "J0ratio" in kwargs:
                        if not len(kwargs["n"]) == len(kwargs["J0ratio"]):
                            raise ValueError("n and J0ratio must be same size")

                    # if only n or J0ratio is set, check if it matches current diode configuration
                    elif not len(value) == len(self.n) and not len(value) == len(self.J0ratio):
                        raise ValueError("setting single n or J0ratio value must match previous number of diodes")

                    self.__dict__[key] = np.array(value)
                    # with self.debugout:
                    #     print("array", key, value)
            elif key in self.ATTR:  # scalar float
                self.__dict__[key] = np.float64(value)

            # raise error if the key is not in the class attributes
            elif key not in list(self.__dict__.keys()):
                raise ValueError(f"invalid class attribute {key}")
            else:
                logger.warning("Junction.set: attribute {!r} exists but is not settable via set(); value ignored", key)

        # Boundary-condition check: the photocurrent property scales as
        # Jphoto = Jext * lightarea / totalarea + JLC, so lightarea must
        # never exceed totalarea (no more than 100 % of the device is
        # illuminated). Validate once after all kwargs are processed so
        # multi-key sets like set(lightarea=X, totalarea=Y) are checked
        # against their final state, not transient intermediates.
        if any(k in kwargs for k in ("area", "lightarea", "totalarea")):
            if self.lightarea > self.totalarea:
                raise ValueError(
                    f"Junction '{self.name}': lightarea ({self.lightarea}) "
                    f"cannot exceed totalarea ({self.totalarea}). "
                    "Jphoto = Jext * lightarea/totalarea assumes the "
                    "illuminated fraction is at most 1."
                )

    @property
    def Jphoto(self) -> float:
        """[A/cm^2] total photocurrent density on the junction.

        Combines the external photocurrent (scaled by the illuminated
        fraction of the total area) and the luminescent-coupling current
        injected from a neighbouring junction:

            Jphoto = Jext * (lightarea / totalarea) + JLC.

        The area scaling assumes Jext is referenced to lightarea and
        spreads it uniformly across totalarea so shaded regions
        contribute zero photocurrent.
        """
        return self.Jext * self.lightarea / self.totalarea + self.JLC

    @property
    def TK(self) -> float:
        """[K] junction temperature in Kelvin (derived from TC)."""
        return TK(self.TC)

    @property
    def Vth(self) -> float:
        """[V] thermal voltage kT/q at the current junction temperature."""
        return Vth(self.TC)

    @property
    def Jdb(self) -> float:
        """[A/cm^2] radiative (detailed-balance) saturation current density.

        Computed via the Rau et al. formulation. This is a thermodynamic quantity (not a free
        parameter) and forms the irreducible lower bound on J0.
        """
        # ``self.theta`` falls back to the class-level default 2.0 for legacy pickled
        # Junctions that predate the generalized-Urbach attribute, preserving the
        # historical Mattheis-Rau-Werner behaviour for those instances.
        return Jdb(self.TC, self.Eg, self.sigma, theta=self.theta)

    @property
    def J0(self) -> np.ndarray:
        """[A/cm^2] per-diode saturation current densities [J0(n0), J0(n1), ...].

        Recomputed on every access from Jdb, n, and
        J0ratio using the formula

            J0[i] = J0_REFERENCE * J0ratio[i]
                    * (Jdb / J0_REFERENCE)^(1/n[i])

        where J0_REFERENCE = 1e-3 A/cm^2 makes the power-law argument and
        J0ratio dimensionless. All currents remain in A/cm^2, with a log-domain
        fallback for numerical stability. Because Jdb depends on temperature,
        J0 automatically tracks changes in TC.
        """

        if (isinstance(self.n, np.ndarray)) and (isinstance(self.J0ratio, np.ndarray)):
            if self.n.size == self.J0ratio.size:
                return _saturation_current_from_ratio(self.Jdb, self.n, self.J0ratio)
            else:
                return np.array(np.nan, dtype=np.float64)  # different sizes
        else:
            return np.array(np.nan, dtype=np.float64)  # not numpy.ndarray

    def _J0init(self, J0ref: list[float] | np.ndarray):
        """
        initialize self.J0ratio from J0ref
        """
        J0ref = np.array(J0ref)
        if self.n.size == J0ref.size:
            self.J0ratio = _ratio_from_saturation_current(self.Jdb, self.n, J0ref)
        else:
            raise ValueError("J0ref and n must be same size")

    def _solver_state(self) -> dict:
        """Snapshot of all junction parameters the solvers need, as plain floats/arrays.

        The J0/Vth/Jdb properties are recomputed on every access; hoisting them
        into this snapshot once per solve (instead of once per residual
        evaluation) removes the dominant cost from the root-finding loops.
        The diode filter matches Jmultidiodes: entries with n <= 0
        or non-finite J0 are dropped.
        """
        n = np.atleast_1d(np.asarray(self.n, dtype=np.float64))
        J0 = np.atleast_1d(np.asarray(self.J0, dtype=np.float64))
        good = (n > 0.0) & np.isfinite(J0)
        Vthf = float(self.Vth)
        Jdbf = float(self.Jdb)
        rbb = self.RBB_dict.copy()
        if rbb.get("method") == "JFG" and rbb["mrb"] != 0.0:
            rbb["J0rb_effective"] = float(_saturation_current_from_ratio(Jdbf, rbb["mrb"], rbb["J0rb"]))
        return {
            "n": n[good],
            "nVth": n[good] * Vthf,
            "J0": J0[good],
            "Vth": Vthf,
            "Gsh": float(self.Gsh),
            "Rser": float(self.Rser),
            "Jdb": Jdbf,
            "gamma": float(self.gamma),
            "notdiode": bool(self.notdiode()),
            "rbb": rbb,
            # plain-python twins for the fast scalar residual
            "J0f": [float(x) for x in J0[good]],
            "ninv": [1.0 / (float(ni) * Vthf) for ni in n[good]],
            "J0sum": float(J0[good].sum()),
        }

    def _solver_state_rows(self, TC, Eg, sigma) -> dict:
        """Per-row twin of _solver_state for arrays of (TC, Eg, sigma).

        Everything that depends on temperature or bandgap gets a leading row
        axis: ``J0`` and ``nVth`` are (rows, diodes), ``Vth``, ``Jdb`` and
        ``notdiode`` are (rows,). The state-based kernels (_recomb_current,
        _recomb_current_deriv, _jem_arr) take this state unchanged when ``V``
        holds one voltage per row. Diodes with n <= 0 are dropped and
        non-finite J0 entries carry no current, as in _solver_state.

        A subclass that redefines J0, Jdb or Vth is evaluated row by row
        through its own properties, so a custom temperature model is honoured.
        """
        TC = np.atleast_1d(np.asarray(TC, dtype=np.float64))
        Eg = np.broadcast_to(np.asarray(Eg, dtype=np.float64), TC.shape)
        sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), TC.shape)
        n = np.atleast_1d(np.asarray(self.n, dtype=np.float64))
        cls = type(self)
        if cls.J0 is Junction.J0 and cls.Jdb is Junction.Jdb and cls.Vth is Junction.Vth:
            Vthr = Vth(TC)
            Jdbr = _jdb_rows(TC, Eg, sigma, float(self.theta))
            ratio = np.atleast_1d(np.asarray(self.J0ratio, dtype=np.float64))
            J0 = _saturation_current_from_ratio(Jdbr[:, None], n[None, :], ratio[None, :])
        else:
            tmp = self.copy()
            Vthr = np.empty(TC.size)
            Jdbr = np.empty(TC.size)
            J0 = np.empty((TC.size, n.size))
            for i in range(TC.size):
                tmp.set(TC=TC[i], Eg=Eg[i], sigma=sigma[i])
                Vthr[i] = tmp.Vth
                Jdbr[i] = tmp.Jdb
                J0[i] = tmp.J0
        with np.errstate(invalid="ignore"):
            notdiode = (self.pn == 0) | (J0.sum(axis=-1) == 0.0)
        keep = n > 0.0
        J0 = np.where(np.isfinite(J0[:, keep]), J0[:, keep], 0.0)
        rbb = self.RBB_dict.copy()
        if rbb.get("method") == "JFG" and rbb["mrb"] != 0.0:
            rbb["J0rb_effective"] = _saturation_current_from_ratio(Jdbr, rbb["mrb"], rbb["J0rb"])
        return {
            "n": n[keep],
            "nVth": n[keep][None, :] * Vthr[:, None],
            "J0": J0,
            "Vth": Vthr,
            "Gsh": float(self.Gsh),
            "Rser": float(self.Rser),
            "Jdb": Jdbr,
            "gamma": float(self.gamma),
            "notdiode": notdiode,
            "rbb": rbb,
        }

    def _vdiode_arr(self, Jtot: np.ndarray, state: dict | None = None) -> np.ndarray:
        """Solve Jtot - _recomb_current(V) = 0 for each element of ``Jtot`` [A/cm^2].

        Vectorized core of Vdiode (no Rser, no Jphoto added here).
        Returns the junction-frame voltages; np.nan where no root is bracketed
        in [-VLIM_REVERSE, VLIM_FORWARD] (matching the historical brentq
        ValueError -> nan behaviour).

        ``state`` is the junction's own operating condition, shared by all
        elements (_solver_state, the default), or a per-row state
        (_solver_state_rows) with one element of ``Jtot`` per row. The solver
        is chosen by SOLVER; a per-row state is always solved by Newton.
        """
        st = self._solver_state() if state is None else state
        Jtot = np.asarray(Jtot, dtype=np.float64)
        if st["J0"].ndim == 2:  # per-row state: element k at the operating condition of row k
            return np.where(st["notdiode"], 0.0, _vdiode_rows(Jtot, st))
        if st["notdiode"]:
            return np.zeros_like(Jtot)

        if not _legacy_solver():
            if Jtot.size >= SCALAR_SOLVE_CUTOFF:
                return _vdiode_rows(Jtot.ravel(), st).reshape(Jtot.shape)
            out = np.empty_like(Jtot)
            for k in range(Jtot.size):
                out.flat[k] = _vjunction_scalar(float(Jtot.flat[k]), st)
            return out

        if Jtot.size < SCALAR_SOLVE_CUTOFF:
            out = np.empty_like(Jtot)
            for k in range(Jtot.size):
                Jt = Jtot.flat[k]
                if not np.isfinite(Jt):
                    out.flat[k] = np.nan
                    continue
                try:
                    out.flat[k] = brentq(lambda V: Jt - _recomb_current_scalar(V, st), -VLIM_REVERSE, VLIM_FORWARD, xtol=XTOL_SOLVE, rtol=EPSREL, maxiter=MAXITER)
                except ValueError:
                    out.flat[k] = np.nan
            return out

        res = elementwise.find_root(
            lambda V, Jt: Jt - _recomb_current(V, st),
            (np.full_like(Jtot, -VLIM_REVERSE), np.full_like(Jtot, VLIM_FORWARD)),
            args=(Jtot,),
        )
        return np.where(res.success, res.x, np.nan)

    def _vmid_arr(self, Vtot: np.ndarray, Jphoto: float | np.ndarray, state: dict | None = None, Rser: float | None = None) -> np.ndarray:
        """Solve Vtot - V + (Jphoto - _recomb_current(V)) * Rser = 0 elementwise.

        Vectorized core of Vmid. ``Jphoto`` may be an array (e.g.
        per-point luminescent coupling). ``Rser`` overrides the junction's own
        series resistance (used by Tandem3T to fold Rz in).
        Returns np.nan where no root is bracketed.

        ``state`` is the junction's own operating condition or a per-row
        state with one element of ``Vtot`` per row, as in _vdiode_arr.
        """
        st = self._solver_state() if state is None else state
        Vtot = np.asarray(Vtot, dtype=np.float64)
        Rs = st["Rser"] if Rser is None else float(Rser)
        if st["J0"].ndim == 2:  # per-row state: element k at the operating condition of row k
            return np.where(st["notdiode"], 0.0, _vmid_rows(Vtot, Jphoto, st, Rs))
        if st["notdiode"]:
            return np.zeros_like(Vtot)
        Jph = np.broadcast_to(np.asarray(Jphoto, dtype=np.float64), Vtot.shape)

        if not _legacy_solver():
            if Vtot.size >= SCALAR_SOLVE_CUTOFF:
                return _vmid_rows(Vtot.ravel(), Jph.ravel(), st, Rs).reshape(Vtot.shape)
            out = np.empty_like(Vtot)
            for k in range(Vtot.size):
                Vt = float(Vtot.flat[k])
                Jp = float(Jph.flat[k])
                if Rs == 0.0:
                    out.flat[k] = Vt if (-VLIM_REVERSE <= Vt <= VLIM_FORWARD and math.isfinite(Jp)) else np.nan
                else:
                    out.flat[k] = _vjunction_scalar(Jp, st, 1.0 / Rs, Vt)
            return out

        if Vtot.size < SCALAR_SOLVE_CUTOFF:
            out = np.empty_like(Vtot)
            for k in range(Vtot.size):
                Vt = Vtot.flat[k]
                Jp = Jph.flat[k]
                if not (np.isfinite(Vt) and np.isfinite(Jp)):
                    out.flat[k] = np.nan
                    continue
                try:
                    out.flat[k] = brentq(lambda V: Vt - V + (Jp - _recomb_current_scalar(V, st)) * Rs, -VLIM_REVERSE, VLIM_FORWARD, xtol=XTOL_SOLVE, rtol=EPSREL, maxiter=MAXITER)
                except ValueError:
                    out.flat[k] = np.nan
            return out

        res = elementwise.find_root(
            lambda V, Vt, Jp: Vt - V + (Jp - _recomb_current(V, st)) * Rs,
            (np.full_like(Vtot, -VLIM_REVERSE), np.full_like(Vtot, VLIM_FORWARD)),
            args=(Vtot, Jph),
        )
        return np.where(res.success, res.x, np.nan)

    def Jem(self, Vmid: float) -> float:
        r"""[A/cm^2] light emitted from the junction (current density).

        Two physically distinct contributions per Lan and Green,
        Appl. Phys. Lett. 106, 263902 (2015), Eqs. 2a-2b:

        * EL (Rau reciprocity): carriers that reach the junction and
          recombine radiatively across the depletion region.  Scales
          as Jdb * (exp(Vmid/Vth) - 1) and vanishes at short circuit.
          Suppressed for Vmid <= 0 because the diode-equation form
          would otherwise describe absorption, not emission.
        * PL (Lan and Green): carriers that recombine radiatively in
          the absorber before reaching the junction.  Scales as
          gamma * Jphoto and is present at every bias, including
          short circuit and reverse bias.  Tayagaki et al. 2018
          (Fig. 5b) shows this nonzero V_top=0 baseline experimentally.

        With the default gamma = 0 the PL term is zero, so Jem
        reduces to the pure-EL form and Jem(Vmid <= 0) == 0.
        """
        # PL contribution is voltage-independent (still active at V <= 0).
        Jem = self.gamma * self.Jphoto  # PL Lan and Green
        # EL contribution only above short circuit.
        if Vmid > 0.0:
            Jem += self.Jdb * (np.exp(Vmid / self.Vth) - 1.0)  # EL Rau
        return Jem

    def notdiode(self) -> bool:
        """
        is this junction really a diode
        or just a resistor
        sum(J0) = 0 -> not diode
        pn = 0 -> not diode
        """
        if self.pn == 0:
            return True

        jsum = np.float64(0.0)
        for saturation_current in self.J0:
            jsum += saturation_current

        return jsum == np.float64(0.0)

    def Jmultidiodes(self, Vdiode: float) -> float:
        """
        calculate recombination current density from
        multiple diodes self.n, self.J0 numpy.ndarray
        two-diodes:
        n  = [1, 2]  #two diodes
        J0 = [10,10]  #poor cell
        detailed balance:
        n  = [1]
        J0 = [1]
        three-diodes
        n = [1, 1.8, (2/3)]
        """
        Jrec = np.float64(0.0)
        for ideality_factor, saturation_current in zip(self.n, self.J0):
            if ideality_factor > 0.0 and math.isfinite(saturation_current):
                # try:
                Jrec += saturation_current * (np.exp(Vdiode / self.Vth / ideality_factor) - 1.0)
                # except ValueError:
                # continue

        return Jrec

    def JshuntRBB(self, Vdiode: float) -> float:
        """
        return shunt + reverse-bias breakdown current

            RBB_dict={'method':None}   #None

            RBB_dict={'method':'JFG', mrb'':10., 'J0rb':1., 'Vrb':0.}

            RBB_dict={'method':'bishop','mrb'':3.28, 'avalanche':1, 'Vrb':-5.5}

            RBB_dict={'method':'pvmismatch','ARBD':arbd,'BRBD':brbd,'VRBD':vrb,'NRBD':nrbd:

        Vdiode without Rs
        Vth = kT
        Gshunt
        """

        RBB_dict = self.RBB_dict
        method = RBB_dict["method"]
        JRBB = np.float64(0.0)

        if method == "JFG":
            Vrb = RBB_dict["Vrb"]
            J0rb = RBB_dict["J0rb"]
            mrb = RBB_dict["mrb"]
            if Vdiode <= Vrb and mrb != 0.0:
                J0rb_effective = _saturation_current_from_ratio(self.Jdb, mrb, J0rb)
                JRBB = -J0rb_effective * (np.exp(-Vdiode / self.Vth / mrb) - 1.0)

        elif method == "bishop":
            Vrb = RBB_dict["Vrb"]
            a = RBB_dict["avalanche"]
            mrb = RBB_dict["mrb"]
            if Vdiode <= 0.0 and Vrb != 0.0:
                JRBB = Vdiode * self.Gsh * a * (1.0 + Vdiode / Vrb) ** (-mrb)

        elif method == "pvmismatch":
            raise NotImplementedError("RBB method 'pvmismatch' is documented but not implemented. Use RBB='JFG', RBB='bishop', or RBB=None.")

        # else:
        #     JRBB = self.J0.sum()

        return Vdiode * self.Gsh + JRBB

    def Jparallel(self, Vdiode: float, Jtot: float) -> float:
        """
        circuit equation to be zeroed to solve for Vi
        for voltage across parallel diodes with shunt and reverse breakdown
        """

        if self.notdiode():  # sum(J0)=0 -> no diode
            return Jtot

        JLED = self.Jmultidiodes(Vdiode)
        JRBB = self.JshuntRBB(Vdiode)
        # JRBB = JshuntRBB(Vdiode, self.Vth, self.Gsh, self.RBB_dict)
        return Jtot - JLED - JRBB

    def Vdiode(self, Jdiode: float | np.ndarray) -> float | np.ndarray:
        """
        Jtot = Jphoto + J
        for junction self of class Junction
        return Vdiode(Jtot)
        no Rseries here

        ``Jdiode`` may be a scalar (returns float, historical behaviour) or an
        ndarray (returns an ndarray of the same shape, solved in one
        vectorized call).
        """
        st = self._solver_state()
        scalar = np.ndim(Jdiode) == 0
        if st["notdiode"]:  # sum(J0)=0 -> no diode
            return 0.0 if scalar else np.zeros(np.shape(Jdiode))

        Jtot = self.Jphoto + np.asarray(Jdiode, dtype=np.float64)
        out = self._vdiode_arr(np.atleast_1d(Jtot).ravel(), state=st)
        if scalar:
            return float(out[0])
        return out.reshape(np.shape(Jdiode))

    def _dV(self, Vmid: float, Vtot: float) -> float:
        """
        see singlejunction
        circuit equation to be zeroed (returns voltage difference) to solve for Vmid
        single junction circuit with series resistance and parallel diodes
        """

        J = self.Jparallel(Vmid, self.Jphoto)
        dV = Vtot - Vmid + J * self.Rser
        return dV

    def Vmid(self, Vtot: float | np.ndarray) -> float | np.ndarray:
        """
        see Vparallel
        find intermediate voltage in a single junction diode with series resistance
        Given Vtot=Vparallel + Rser * Jparallel

        ``Vtot`` may be a scalar (returns float, historical behaviour) or an
        ndarray (returns an ndarray of the same shape, solved in one
        vectorized call).
        """
        st = self._solver_state()
        scalar = np.ndim(Vtot) == 0
        if st["notdiode"]:  # sum(J0)=0 -> no diode
            return 0.0 if scalar else np.zeros(np.shape(Vtot))

        out = self._vmid_arr(np.atleast_1d(np.asarray(Vtot, dtype=np.float64)).ravel(), self.Jphoto, state=st)
        if scalar:
            return float(out[0])
        return out.reshape(np.shape(Vtot))

    # def controls(self):
    #     """
    #     use interactive_output for GUI in IPython
    #     """

    #     cell_layout = widgets.Layout(display="inline_flex", flex_flow="row", justify_content="flex-end", width="300px")
    #     # controls
    #     in_name = widgets.Text(value=self.name, description="name", layout=cell_layout, continuous_update=False)
    #     in_Eg = widgets.FloatSlider(
    #         value=self.Eg, min=0.1, max=3.0, step=0.01, description="Eg", layout=cell_layout, readout_format=".2f"
    #     )
    #     in_TC = widgets.FloatSlider(
    #         value=self.TC, min=-40, max=200.0, step=2, description="TC", layout=cell_layout, readout_format=".1f"
    #     )
    #     in_Jext = widgets.FloatSlider(
    #         value=self.Jext, min=0.0, max=0.080, step=0.001, description="Jext", layout=cell_layout, readout_format=".4f"
    #     )
    #     in_JLC = widgets.FloatSlider(
    #         value=self.JLC,
    #         min=0.0,
    #         max=0.080,
    #         step=0.001,
    #         description="JLC",
    #         layout=cell_layout,
    #         readout_format=".4f",
    #         disabled=True,
    #     )
    #     in_Gsh = widgets.FloatLogSlider(
    #         value=self.Gsh, base=10, min=-12, max=3, step=0.01, description="Gsh", layout=cell_layout, readout_format=".2e"
    #     )
    #     in_Rser = widgets.FloatLogSlider(
    #         value=self.Rser, base=10, min=-7, max=3, step=0.01, description="Rser", layout=cell_layout, readout_format=".2e"
    #     )
    #     in_lightarea = widgets.FloatLogSlider(
    #         value=self.lightarea, base=10, min=-6, max=3.0, step=0.1, description="lightarea", layout=cell_layout
    #     )
    #     in_totalarea = widgets.FloatSlider(
    #         value=self.totalarea, min=self.lightarea, max=1e3, step=0.1, description="totalarea", layout=cell_layout
    #     )
    #     in_beta = widgets.FloatSlider(
    #         value=self.beta, min=0.0, max=50.0, step=0.1, description="beta", layout=cell_layout, readout_format=".2e"
    #     )
    #     in_gamma = widgets.FloatSlider(
    #         value=self.gamma, min=0.0, max=3.0, step=0.1, description="gamma", layout=cell_layout, readout_format=".2e"
    #     )
    #     in_pn = widgets.IntSlider(value=self.pn, min=-1, max=1, step=1, description="pn", layout=cell_layout)

    #     # linkages
    #     # arealink = widgets.jslink((in_lightarea, "value"), (in_totalarea, "min"))  # also jsdlink works

    #     # attr = ["name"] + self.ATTR.copy()
    #     cntrls = [in_name, in_Eg, in_TC, in_Gsh, in_Rser, in_lightarea, in_totalarea, in_Jext, in_JLC, in_beta, in_gamma, in_pn]
    #     # sing_dict = dict(zip(attr, cntrls))
    #     # singout = widgets.interactive_output(self.set, sing_dict)  #all at once

    #     def on_juncchange(change):
    #         # function for changing values
    #         old = change["old"]  # old value
    #         new = change["new"]  # new value
    #         owner = change["owner"]  # control
    #         value = owner.value
    #         desc = owner.description

    #         if new == old:
    #             with self.debugout:
    #                 print("Jcontrol: " + desc + "=", value)
    #         else:
    #             with self.debugout:
    #                 print("Jcontrol: " + desc + "->", value)
    #             self.set(**{desc: value})

    #         # iout.clear_output()
    #         # with iout: print(self)

    #     # diode array
    #     in_tit = widgets.Label(value="Junction", description="Junction")
    #     in_diodelab = widgets.Label(value="diodes:", description="diodes:")
    #     # diode_layout = widgets.Layout(flex_flow="column", align_items="center")

    #     cntrls.append(in_diodelab)
    #     in_n = []  # empty list of n controls
    #     in_ratio = []  # empty list of Jratio controls
    #     diode_dict = {}
    #     for i in range(len(self.n)):
    #         in_n.append(
    #             widgets.FloatLogSlider(
    #                 value=self.n[i], base=10, min=-1, max=1, step=0.001, description="n[" + str(i) + "]", layout=cell_layout
    #             )
    #         )
    #         in_ratio.append(
    #             widgets.FloatLogSlider(
    #                 value=self.J0ratio[i],
    #                 base=10,
    #                 min=-6,
    #                 max=6,
    #                 step=0.1,
    #                 description="J0ratio[" + str(i) + "]",
    #                 layout=cell_layout,
    #             )
    #         )
    #         cntrls.append(in_n[i])
    #         cntrls.append(in_ratio[i])
    #         diode_dict["n[" + str(i) + "]"] = in_n[i]
    #         diode_dict["J0ratio[" + str(i) + "]"] = in_ratio[i]
    #         # hui.append(widgets.HBox([in_n[i],in_ratio[i]]))
    #         # cntrls.append(hui[i])

    #     # diodeout = widgets.interactive_output(self.set, diode_dict)  #all at once

    #     if self.RBB_dict:
    #         RBB_keys = list(self.RBB_dict.keys())
    #         in_rbblab = widgets.Label(value="RBB:", description="RBB:")
    #         cntrls.append(in_rbblab)
    #         in_rbb = []  # empty list of n controls
    #         for i, key in enumerate(RBB_keys):
    #             with self.debugout:
    #                 print("RBB:", i, key)
    #             if key == "method":
    #                 in_rbb.append(
    #                     widgets.Dropdown(
    #                         options=["", "JFG", "bishop"],
    #                         value=self.RBB_dict[key],
    #                         description=key,
    #                         layout=cell_layout,
    #                         continuous_update=False,
    #                     )
    #                 )
    #             else:
    #                 in_rbb.append(
    #                     widgets.FloatLogSlider(
    #                         value=self.RBB_dict[key], base=10, min=-10, max=5, step=0.1, description=key, layout=cell_layout
    #                     )
    #                 )
    #             cntrls.append(in_rbb[i])

    #     for cntrl in cntrls:
    #         cntrl.observe(on_juncchange, names="value")

    #     # output
    #     iout = widgets.Output()
    #     iout.layout.height = "5px"
    #     # with iout: print(self)
    #     cntrls.append(iout)

    #     # user interface
    #     box_layout = widgets.Layout(
    #         display="flex", flex_flow="column", align_items="center", border="1px solid black", width="320px", height="350px"
    #     )

    #     ui = widgets.VBox([in_tit] + cntrls, layout=box_layout)
    #     self.ui = ui  # make it an attribute

    #     return ui
