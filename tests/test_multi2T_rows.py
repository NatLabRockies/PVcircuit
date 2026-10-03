# noqa: N999
"""
Tests for Multi2T.V2T_rows / MPP_rows: many operating conditions solved at
once must reproduce the device solved row by row.
"""

import contextlib

import numpy as np
import pytest

import pvcircuit as pvc
from pvcircuit import Multi2T

KEYS = ("Voc", "Isc", "Vmp", "Imp", "Pmp")


@contextlib.contextmanager
def _solver(name):
    """Run a block with pvcircuit.junction.SOLVER set to ``name``."""
    previous = pvc.junction.SOLVER
    pvc.junction.SOLVER = name
    try:
        yield
    finally:
        pvc.junction.SOLVER = previous


def _rows(njuncs=2, nrows=12, seed=0):
    """Operating conditions with temperature, bandgap and photocurrent spread, including strong mismatch."""
    rng = np.random.default_rng(seed)
    TC = np.linspace(-10.0, 80.0, nrows)
    dEg = -4e-4 * (TC - 25.0)  # [eV] bandgap shift with temperature
    Jext = rng.uniform(0.002, 0.020, (nrows, 1)) * rng.uniform(0.6, 1.4, (nrows, njuncs))  # [A/cm^2]
    return TC, dEg, Jext


def _inputs(model, sigma=0.0, seed=0):
    TC, dEg, Jext = _rows(model.njuncs, seed=seed)
    Eg = np.array([float(junc.Eg) for junc in model.j])[None, :] + dEg[:, None]
    return TC, Eg, np.full_like(Eg, sigma), Jext


def _per_row(model, TC, Eg, sigma, Jext, solver="legacy"):
    """Reference: Multi2T.MPP, one row at a time, by default with the legacy solver."""
    out = {key: np.empty(len(TC)) for key in KEYS}
    dev = model.copy()
    with _solver(solver):
        for i in range(len(TC)):
            for k, junc in enumerate(dev.j):
                junc.set(Eg=Eg[i, k], sigma=sigma[i, k], Jext=Jext[i, k], TC=TC[i])
            mpp = dev.MPP()
            for key in KEYS:
                out[key][i] = mpp[key]
    return out


def _assert_matches(rows, ref):
    np.testing.assert_allclose(rows["Voc"], ref["Voc"], rtol=1e-9)
    # the legacy Multi2T.Isc steps to the root with a relative resolution of 1e-7
    np.testing.assert_allclose(rows["Isc"], ref["Isc"], rtol=1e-6)
    # Multi2T.MPP stops on a grid: its Pmp is up to ~2e-7 low, Imp/Vmp resolve ~2e-4
    np.testing.assert_allclose(rows["Pmp"], ref["Pmp"], rtol=1e-6)
    assert np.all(rows["Pmp"] >= ref["Pmp"] * (1.0 - 1e-9)), "the refined optimum must not be below the grid optimum"
    np.testing.assert_allclose(rows["Vmp"], ref["Vmp"], rtol=2e-3)
    np.testing.assert_allclose(rows["Imp"], ref["Imp"], rtol=2e-3)


def _tweak(model, **kwargs):
    dev = model.copy()
    for junc in dev.j:
        junc.set(**kwargs)
    return dev


def _shunt_rs():
    dev = _tweak(Multi2T(), Gsh=1e-3)
    dev.set(Rs2T=3.0)
    return dev


def _unequal_areas():
    dev = Multi2T()
    dev.j[0].set(totalarea=0.8, lightarea=0.8)
    dev.j[1].set(lightarea=0.9)
    return dev


def _resistor_top():
    dev = Multi2T()
    dev.j[0].set(pn=0)  # not a diode: adds no voltage, still hands its photoluminescence down
    return dev


DEVICES = {
    "default": lambda: Multi2T(),
    "no_LC": lambda: _tweak(Multi2T(), beta=0.0),
    "shunt_and_Rs2T": _shunt_rs,
    "very_leaky": lambda: _tweak(Multi2T(), Gsh=2e-2),
    "PL_coupling": lambda: _tweak(Multi2T(), gamma=0.5),
    "RBB_JFG": lambda: _tweak(Multi2T(), RBB="JFG"),
    "RBB_bishop": lambda: _tweak(_tweak(Multi2T(), RBB="bishop"), Gsh=1e-3),
    "three_diodes": lambda: _tweak(Multi2T(), n=[1.0, 2.0, 0.6667], J0ratio=[10.0, 10.0, 0.01]),
    "unequal_areas": _unequal_areas,
    "resistor_top": _resistor_top,
    "three_junctions": lambda: Multi2T(Eg_list=[1.9, 1.4, 1.0]),
    "single_junction": lambda: Multi2T.from_single_junction(pvc.Junction(Eg=1.12, Rser=0.5)),
}


@pytest.mark.parametrize("name", DEVICES)
def test_mpp_rows_matches_mpp(name):
    model = DEVICES[name]()
    TC, Eg, sigma, Jext = _inputs(model)
    rows = model.MPP_rows(TC, Eg, sigma, Jext)
    _assert_matches(rows, _per_row(model, TC, Eg, sigma, Jext))
    # the fast per-row solver finds the same short circuit to its own tolerance
    fast = _per_row(model, TC, Eg, sigma, Jext, solver="fast")
    np.testing.assert_allclose(rows["Voc"], fast["Voc"], rtol=1e-11)
    np.testing.assert_allclose(rows["Isc"], fast["Isc"], rtol=1e-9)


@pytest.mark.parametrize("theta, sigma", [(2.0, 0.03), (1.0, 0.015)])
def test_mpp_rows_matches_mpp_with_band_tails(theta, sigma):
    model = _tweak(Multi2T(), theta=theta)
    TC, Eg, sig, Jext = _inputs(model, sigma=sigma)
    _assert_matches(model.MPP_rows(TC, Eg, sig, Jext), _per_row(model, TC, Eg, sig, Jext))


def test_mpp_rows_follows_the_search_of_mpp_on_a_kinked_power_curve():
    """Weak, mismatched light on a shunted stack with Bishop breakdown: the power curve has two
    nearly equal maxima. MPP_rows scans like MPP, so it must not settle on the lower one."""
    model = _tweak(_tweak(Multi2T(), RBB="bishop"), Gsh=1e-3)
    TC = np.array([7.1, 7.1])
    Eg = np.array([[1.8072, 1.4072], [1.8072, 1.4072]])
    Jext = np.array([[0.11193518e-3, 0.3055661e-3], [0.3e-3, 0.11e-3]])
    sigma = np.zeros_like(Eg)
    rows = model.MPP_rows(TC, Eg, sigma, Jext)
    ref = _per_row(model, TC, Eg, sigma, Jext)
    assert np.all(rows["Pmp"] >= ref["Pmp"] * (1.0 - 1e-9))
    np.testing.assert_allclose(rows["Pmp"], ref["Pmp"], rtol=1e-6)


def test_v2t_rows_matches_v2t_including_no_solution():
    """V2T_rows agrees row by row; beyond the limiting current both return nan."""
    model = Multi2T()  # no shunt: the limiting junction saturates
    TC, Eg, sigma, Jext = _inputs(model)
    for frac in (0.0, 0.5, 0.9, 1.5):
        current = -frac * Jext.min(axis=1)
        V = model.V2T_rows(current, TC, Eg, sigma, Jext)
        dev = model.copy()
        with _solver("legacy"):
            for i in range(len(TC)):
                for k, junc in enumerate(dev.j):
                    junc.set(Eg=Eg[i, k], sigma=sigma[i, k], Jext=Jext[i, k], TC=TC[i])
                np.testing.assert_allclose(V[i], dev.V2T(current[i]), rtol=1e-9, atol=1e-10, equal_nan=True)
    assert np.all(np.isnan(model.V2T_rows(-1.5 * Jext.min(axis=1), TC, Eg, sigma, Jext)))


def test_v2t_rows_broadcasts_scalars():
    """A scalar current, and a scalar sigma, apply to every row."""
    model = Multi2T()
    TC, Eg, sigma, Jext = _inputs(model)
    Voc = model.V2T_rows(0.0, TC, Eg, 0.0, Jext)
    np.testing.assert_array_equal(Voc, model.V2T_rows(np.zeros(len(TC)), TC, Eg, sigma, Jext))
    np.testing.assert_array_equal(Voc, model.MPP_rows(TC, Eg, 0.0, Jext)["Voc"])
    assert Voc.shape == TC.shape


def test_states_are_taken_per_point():
    """The MPP_rows scan evaluates point k with the state of row index[k]."""
    model = Multi2T()
    TC, Eg, sigma, Jext = _inputs(model)
    _, states, Jphoto0 = model._state_rows(TC, Eg, sigma, Jext)
    index = np.array([3, 3, 7, 0])
    current = -np.array([0.2, 0.6, 0.4, 0.0]) * Jext.min(axis=1)[index]
    V = model._V2T_states(current, [pvc.junction._state_take(st, index) for st in states], [Jph[index] for Jph in Jphoto0])
    for k, row in enumerate(index):
        single = model.V2T_rows(current[k], TC[row : row + 1], Eg[row : row + 1], sigma[row : row + 1], Jext[row : row + 1])
        np.testing.assert_allclose(V[k], single[0], rtol=1e-12)


class _ActivatedJ0(pvc.Junction):
    """Junction with an extra thermal activation of J0 on top of the pvcircuit model."""

    @property
    def J0(self):
        return pvc.Junction.J0.fget(self) * np.exp(0.2 * (1.0 / pvc.junction.Vth(25.0) - 1.0 / self.Vth))


def test_junction_subclass_with_own_J0_is_honoured():
    model = Multi2T()
    model.j[0].__class__ = _ActivatedJ0
    TC, Eg, sigma, Jext = _inputs(model)
    rows = model.MPP_rows(TC, Eg, sigma, Jext)
    _assert_matches(rows, _per_row(model, TC, Eg, sigma, Jext))
    plain = Multi2T().MPP_rows(TC, Eg, sigma, Jext)
    assert np.max(np.abs(rows["Voc"] - plain["Voc"])) > 1e-3, "the activation must change the result"


def test_rows_without_photocurrent_are_nan_not_wrong():
    model = Multi2T()
    TC, Eg, sigma, Jext = _inputs(model)
    Jext[4] = 0.0
    mpp = model.MPP_rows(TC, Eg, sigma, Jext)
    assert not np.isfinite(mpp["Pmp"][4]) or mpp["Pmp"][4] == 0.0
    keep = np.arange(len(TC)) != 4
    ref = _per_row(model, TC[keep], Eg[keep], sigma[keep], Jext[keep])
    _assert_matches({key: mpp[key][keep] for key in KEYS}, ref)


def test_bottom_junction_fed_by_luminescent_coupling_only():
    """No external light on the bottom junction: its current is the LC from the top junction."""
    model = Multi2T()
    TC, Eg, sigma, Jext = _inputs(model)
    Jext[:, 1] = 0.0
    rows = model.MPP_rows(TC, Eg, sigma, Jext)
    assert np.all(rows["Pmp"] > 0.0)
    _assert_matches(rows, _per_row(model, TC, Eg, sigma, Jext))


def test_rows_do_not_modify_the_device():
    model = Multi2T()
    before = str(model)
    Vmid = model.Vmid.copy()
    TC, Eg, sigma, Jext = _inputs(model)
    model.MPP_rows(TC, Eg, sigma, Jext)
    model.V2T_rows(0.0, TC, Eg, sigma, Jext)
    assert str(model) == before
    np.testing.assert_array_equal(model.Vmid, Vmid)
