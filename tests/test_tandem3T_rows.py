# noqa: N999
"""
Tests for the Tandem3T *_rows methods: many operating conditions solved at
once must reproduce the device solved row by row.
"""

import contextlib

import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

import pvcircuit as pvc
from pvcircuit import IV3T, Tandem3T

DEVKEYS = ("Iro", "Izo", "Ito", "Vzt", "Vrz", "Vtr")


@contextlib.contextmanager
def _solver(name):
    """Run a block with pvcircuit.junction.SOLVER set to ``name``."""
    previous = pvc.junction.SOLVER
    pvc.junction.SOLVER = name
    try:
        yield
    finally:
        pvc.junction.SOLVER = previous


def _r_type():
    dev = Tandem3T()
    dev.bot.pn = dev.top.pn  # equal pn: junctions reversed against each other
    return dev


def _r_type_no_rz():
    dev = _r_type()
    dev.set(Rz=0)
    return dev


def _no_rz():
    dev = Tandem3T()
    dev.set(Rz=0)
    return dev


def _leaky():
    dev = Tandem3T()
    dev.set(Gsh=1e-3, Rser=0.5)
    return dev


def _unequal_areas():
    dev = Tandem3T()
    dev.top.set(totalarea=0.8, lightarea=0.8)
    dev.bot.set(lightarea=0.9)
    return dev


def _three_diodes_breakdown():
    dev = Tandem3T()
    dev.set(n=[1.0, 2.0, 0.6667], J0ratio=[10.0, 10.0, 0.01], RBB="JFG")
    return dev


DEVICES = {
    "default": Tandem3T,
    "no_Rz": _no_rz,
    "r_type": _r_type,
    "r_type_no_Rz": _r_type_no_rz,
    "leaky_with_Rser": _leaky,
    "reversed_polarity": lambda: Tandem3T(pn=[1, -1]),
    "unequal_areas": _unequal_areas,
    "three_diodes_breakdown": _three_diodes_breakdown,
}


def _inputs(model, nrows=10, sigma=0.0, seed=0):
    """Operating conditions with temperature, bandgap and photocurrent spread, including strong mismatch."""
    rng = np.random.default_rng(seed)
    TC = np.linspace(-10.0, 80.0, nrows)
    dEg = -4e-4 * (TC - 25.0)  # [eV] bandgap shift with temperature
    Jext = rng.uniform(0.002, 0.020, (nrows, 1)) * rng.uniform(0.6, 1.4, (nrows, 2))  # [A/cm^2]
    Eg = np.array([float(model.top.Eg), float(model.bot.Eg)])[None, :] + dEg[:, None]
    return TC, Eg, np.full_like(Eg, sigma), Jext


def _set_row(dev, i, TC, Eg, sigma, Jext):
    dev.top.set(Eg=Eg[i, 0], sigma=sigma[i, 0], Jext=Jext[i, 0], TC=TC[i])
    dev.bot.set(Eg=Eg[i, 1], sigma=sigma[i, 1], Jext=Jext[i, 1], TC=TC[i])


def _per_row(model, TC, Eg, sigma, Jext, solve, solver="legacy"):
    """Reference: ``solve(device)`` returns an IV3T point, one row at a time, by default with the legacy solver."""
    out = {key: np.empty(len(TC)) for key in (*DEVKEYS, "Ptot")}
    dev = model.copy()
    with _solver(solver):
        for i in range(len(TC)):
            _set_row(dev, i, TC, Eg, sigma, Jext)
            pt = solve(dev)
            for key in out:
                out[key][i] = getattr(pt, key)[0]
    return out


@pytest.mark.parametrize("name", DEVICES)
def test_voc3_isc3_rows_match_voc3_isc3(name):
    model = DEVICES[name]()
    TC, Eg, sigma, Jext = _inputs(model)
    voc = model.Voc3_rows(TC, Eg, sigma, Jext)
    ref = _per_row(model, TC, Eg, sigma, Jext, lambda dev: dev.Voc3())
    for key in ("Vzt", "Vrz", "Vtr"):
        np.testing.assert_allclose(getattr(voc, key), ref[key], rtol=0, atol=1e-9)
    isc = model.Isc3_rows(TC, Eg, sigma, Jext)
    ref = _per_row(model, TC, Eg, sigma, Jext, lambda dev: dev.Isc3())
    for key in ("Iro", "Izo", "Ito"):
        np.testing.assert_allclose(getattr(isc, key), ref[key], rtol=1e-8, atol=1e-13)
    assert isc.shape == voc.shape == (len(TC),)


@pytest.mark.parametrize("name", DEVICES)
def test_v3t_and_i3trel_rows_match_v3t_and_i3trel(name):
    """Arbitrary operating points, one per row: same voltages / currents as the device set to that row."""
    model = DEVICES[name]()
    TC, Eg, sigma, Jext = _inputs(model)
    nrows = len(TC)
    rng = np.random.default_rng(1)
    isc = model.Isc3_rows(TC, Eg, sigma, Jext)
    voc = model.Voc3_rows(TC, Eg, sigma, Jext)

    iv = IV3T(name="rows", meastype="CZ", shape=nrows)
    iv.Ito = isc.Ito * rng.uniform(0.2, 0.98, nrows)
    iv.Iro = isc.Iro * rng.uniform(0.2, 0.98, nrows)
    iv.Izo = -(iv.Ito + iv.Iro)
    assert model.V3T_rows(iv, TC, Eg, sigma, Jext) == 0

    iw = IV3T(name="rows", meastype="CZ", shape=nrows)
    iw.Vzt = voc.Vzt * rng.uniform(0.1, 0.9, nrows)
    iw.Vrz = voc.Vrz * rng.uniform(0.1, 0.9, nrows)
    iw.Vtr = -(iw.Vzt + iw.Vrz)
    assert model.I3Trel_rows(iw, TC, Eg, sigma, Jext) == 0

    dev = model.copy()
    with _solver("legacy"):
        for i in range(nrows):
            _set_row(dev, i, TC, Eg, sigma, Jext)
            pt = IV3T(name="pt", meastype="CZ", shape=1)
            pt.set(Iro=iv.Iro[i], Izo=iv.Izo[i], Ito=iv.Ito[i])
            dev.V3T(pt)
            np.testing.assert_allclose([iv.Vzt[i], iv.Vrz[i], iv.Vtr[i], iv.Ptot[i]], [pt.Vzt[0], pt.Vrz[0], pt.Vtr[0], pt.Ptot[0]], rtol=1e-9, atol=1e-9)
            pt = IV3T(name="pt", meastype="CZ", shape=1)
            pt.set(Vzt=iw.Vzt[i], Vrz=iw.Vrz[i], Vtr=iw.Vtr[i])
            dev.I3Trel(pt)
            np.testing.assert_allclose([iw.Iro[i], iw.Izo[i], iw.Ito[i], iw.Ptot[i]], [pt.Iro[0], pt.Izo[0], pt.Ito[0], pt.Ptot[0]], rtol=1e-7, atol=1e-11)


@pytest.mark.parametrize("name", DEVICES)
def test_mpp_rows_matches_mpp(name):
    """Two independent loads: Tandem3T.MPP stops on a coarse current grid, MPP_rows refines to the optimum."""
    model = DEVICES[name]()
    TC, Eg, sigma, Jext = _inputs(model)
    rows = model.MPP_rows(TC, Eg, sigma, Jext)
    ref = _per_row(model, TC, Eg, sigma, Jext, lambda dev: dev.MPP())
    assert np.all(rows.Ptot >= ref["Ptot"] * (1.0 - 1e-9)), "the refined optimum must not be below the grid optimum"
    np.testing.assert_allclose(rows.Ptot, ref["Ptot"], rtol=3e-3)
    for key in ("Iro", "Ito"):
        np.testing.assert_allclose(getattr(rows, key), ref[key], rtol=3e-2)
    for key in ("Vzt", "Vrz"):
        np.testing.assert_allclose(getattr(rows, key), ref[key], rtol=5e-2)
    # the point is a consistent solution of the circuit
    np.testing.assert_allclose(rows.Iro + rows.Izo + rows.Ito, 0.0, atol=1e-15)
    np.testing.assert_allclose(rows.Vzt + rows.Vrz + rows.Vtr, 0.0, atol=1e-12)


@pytest.mark.parametrize("name", ["default", "no_Rz", "r_type", "leaky_with_Rser"])
def test_mpp_rows_is_the_optimum(name):
    """Against a direct maximization of the total power over both currents on the device itself."""
    model = DEVICES[name]()
    TC, Eg, sigma, Jext = _inputs(model, nrows=4, seed=2)
    rows = model.MPP_rows(TC, Eg, sigma, Jext)
    dev = model.copy()
    for i in range(len(TC)):
        _set_row(dev, i, TC, Eg, sigma, Jext)
        isc = dev.Isc3()
        scale = np.array([isc.Ito[0], isc.Iro[0]])

        def negative_power(x, scale=scale):
            pt = IV3T(name="pt", meastype="CZ", shape=1)
            pt.set(Ito=x[0] * scale[0], Iro=x[1] * scale[1], Izo=-(x[0] * scale[0] + x[1] * scale[1]))
            dev.V3T(pt)
            return -pt.Ptot[0]

        best = minimize(negative_power, [0.9, 0.9], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-14, maxiter=2000))
        assert best.success
        np.testing.assert_allclose(rows.Ptot[i], -best.fun, rtol=1e-8)


def test_mpp_rows_reaches_the_optimum_of_a_strongly_top_rich_row():
    """Luminescent coupling makes a ridge along which the alternating search over the two currents zig-zags.

    Regression: after 8 alternations the power was still rising, 3.6e-5 below
    the two-current maximum.
    """
    model = Tandem3T()
    TC = np.array([25.0])
    Eg = np.array([[float(model.top.Eg), float(model.bot.Eg)]])
    sigma = np.array([[float(model.top.sigma), float(model.bot.sigma)]])
    Jext = np.array([[0.060, 0.003]])  # [A/cm^2]
    rows = model.MPP_rows(TC, Eg, sigma, Jext)
    dev = model.copy()
    _set_row(dev, 0, TC, Eg, sigma, Jext)
    isc = dev.Isc3()
    scale = np.array([isc.Ito[0], isc.Iro[0]])

    def negative_power(x):
        pt = IV3T(name="pt", meastype="CZ", shape=1)
        pt.set(Ito=x[0] * scale[0], Iro=x[1] * scale[1], Izo=-(x[0] * scale[0] + x[1] * scale[1]))
        dev.V3T(pt)
        return -pt.Ptot[0]

    start = [rows.Ito[0] / scale[0], rows.Iro[0] / scale[1]]  # the MPP_rows point
    best = minimize(negative_power, start, method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-12, maxiter=2000))
    assert best.success
    np.testing.assert_allclose(rows.Ptot[0], -best.fun, rtol=1e-7)


@pytest.mark.parametrize("name", DEVICES)
@pytest.mark.parametrize("ratio", [(2, 1), (1, 1), (3, 2)])
def test_vm_rows_matches_vm(name, ratio):
    """Voltage-matched line: Tandem3T.VM stops on its grid, VM_rows refines to the optimum."""
    model = DEVICES[name]()
    TC, Eg, sigma, Jext = _inputs(model)
    rows = model.VM_rows(*ratio, TC, Eg, sigma, Jext)
    ref = _per_row(model, TC, Eg, sigma, Jext, lambda dev: dev.VM(*ratio)[1])
    assert np.all(np.isfinite([getattr(rows, key) for key in DEVKEYS])), "every row must be solved"
    assert np.all(rows.Ptot >= ref["Ptot"] * (1.0 - 1e-9)), "the refined optimum must not be below the grid optimum"
    np.testing.assert_allclose(rows.Ptot, ref["Ptot"], rtol=5e-5)
    for key in ("Iro", "Ito", "Vzt", "Vrz"):
        np.testing.assert_allclose(getattr(rows, key), ref[key], rtol=1e-2)
    np.testing.assert_allclose(rows.Izo, ref["Izo"], rtol=0, atol=1e-4)  # the small difference of the two junction currents
    # on the voltage-matched line
    np.testing.assert_allclose(np.abs(rows.Vrz) * ratio[0], np.abs(rows.Vzt) * ratio[1], rtol=1e-12)


@pytest.mark.parametrize("name", ["default", "r_type", "leaky_with_Rser"])
def test_vm_rows_is_the_optimum(name):
    """Against a direct maximization along the voltage-matched line on the device itself."""
    model = DEVICES[name]()
    TC, Eg, sigma, Jext = _inputs(model, nrows=4, seed=2)
    rows = model.VM_rows(2, 1, TC, Eg, sigma, Jext)
    dev = model.copy()
    for i in range(len(TC)):
        _set_row(dev, i, TC, Eg, sigma, Jext)

        def negative_power(Vzt, i=i):
            pt = IV3T(name="pt", meastype="CZ", shape=1)
            Vrz = Vzt * rows.Vrz[i] / rows.Vzt[i]  # the line through the origin and the rows optimum
            pt.set(Vzt=Vzt, Vrz=Vrz, Vtr=-(Vzt + Vrz))
            dev.I3Trel(pt)
            return -pt.Ptot[0] if np.isfinite(pt.Iro[0]) else 0.0

        # the power maximum of an r-type cell sits close to where the line leaves the solvable range
        best = minimize_scalar(negative_power, bounds=sorted((0.7 * rows.Vzt[i], 1.05 * rows.Vzt[i])), method="bounded", options=dict(xatol=1e-12))
        np.testing.assert_allclose(rows.Ptot[i], -best.fun, rtol=1e-8)


@pytest.mark.parametrize("make_model", [Tandem3T, _r_type])
@pytest.mark.parametrize("Jext", [(0.012, 0.0), (0.0, 0.012)])
def test_vm_line_keeps_its_orientation_with_a_dark_junction(make_model, Jext):
    """The orientation of the voltage-matched line is a property of the cell type.

    VM took it from the ratio of the two open-circuit voltages, which is 0 / 0
    or rounding noise when a junction is dark.
    """
    lit = make_model()
    voc = lit.Voc3()
    orientation = np.sign(voc.Vzt[0] / voc.Vrz[0])
    dev = make_model()
    dev.bot.set(beta=0.0)  # no luminescent coupling: the dark junction stays dark
    dev.top.set(Jext=Jext[0])
    dev.bot.set(Jext=Jext[1])
    points = []
    for solver in ("fast", "legacy"):
        with _solver(solver):
            points.append(dev.VM(2, 1)[1])
    rows = dev.VM_rows(2, 1, [25.0], [[float(dev.top.Eg), float(dev.bot.Eg)]], 0.0, [Jext])
    for pt in (*points, rows):
        assert pt.Ptot[0] > 0.0
        assert np.sign(pt.Vzt[0] / pt.Vrz[0]) == orientation
        np.testing.assert_allclose(abs(pt.Vrz[0]) * 2, abs(pt.Vzt[0]) * 1, rtol=1e-12)  # on the voltage-matched line
    np.testing.assert_allclose(points[1].Ptot[0], points[0].Ptot[0], rtol=1e-9)
    np.testing.assert_allclose(rows.Ptot[0], points[0].Ptot[0], rtol=5e-5)


def test_vm_without_open_circuit_point_gives_no_power():
    """A negative photocurrent on an unshunted junction has no open-circuit voltage: VM raised NameError."""
    dev = Tandem3T()
    dev.bot.set(Jext=-0.0005, beta=0.0)
    assert not np.isfinite(dev.Voc3().Vrz[0])
    _, pt = dev.VM(2, 1)
    assert pt.Ptot[0] == 0.0
    rows = dev.VM_rows(2, 1, [25.0], [[1.8, 1.4]], 0.0, [[0.014, -0.0005]])
    assert not np.isfinite(rows.Iro[0])


class _ActivatedJ0(pvc.Junction):
    """Junction with an extra thermal activation of J0 on top of the pvcircuit model."""

    @property
    def J0(self):
        return pvc.Junction.J0.fget(self) * np.exp(0.2 * (1.0 / pvc.junction.Vth(25.0) - 1.0 / self.Vth))


def test_junction_subclass_with_own_J0_is_honoured():
    model = Tandem3T()
    model.top.__class__ = _ActivatedJ0
    TC, Eg, sigma, Jext = _inputs(model)
    voc = model.Voc3_rows(TC, Eg, sigma, Jext)
    ref = _per_row(model, TC, Eg, sigma, Jext, lambda dev: dev.Voc3())
    np.testing.assert_allclose(voc.Vzt, ref["Vzt"], rtol=0, atol=1e-9)
    plain = Tandem3T().Voc3_rows(TC, Eg, sigma, Jext)
    assert np.max(np.abs(voc.Vzt - plain.Vzt)) > 1e-3, "the activation must change the result"
    rows = model.VM_rows(2, 1, TC, Eg, sigma, Jext)
    ref = _per_row(model, TC, Eg, sigma, Jext, lambda dev: dev.VM(2, 1)[1])
    np.testing.assert_allclose(rows.Ptot, ref["Ptot"], rtol=5e-5)


@pytest.mark.parametrize("theta, sigma", [(2.0, 0.03), (1.0, 0.015)])
def test_rows_with_band_tails(theta, sigma):
    model = Tandem3T()
    model.set(theta=theta)
    TC, Eg, sig, Jext = _inputs(model, nrows=6, sigma=sigma)
    rows = model.MPP_rows(TC, Eg, sig, Jext)
    ref = _per_row(model, TC, Eg, sig, Jext, lambda dev: dev.MPP())
    assert np.all(rows.Ptot >= ref["Ptot"] * (1.0 - 1e-9))
    np.testing.assert_allclose(rows.Ptot, ref["Ptot"], rtol=3e-3)


def test_dark_rows_give_zero_power_not_wrong_numbers():
    model = Tandem3T()
    TC, Eg, sigma, Jext = _inputs(model)
    Jext[4] = 0.0
    keep = np.arange(len(TC)) != 4
    for rows, ref in (
        (model.MPP_rows(TC, Eg, sigma, Jext), model.MPP_rows(TC[keep], Eg[keep], sigma[keep], Jext[keep])),
        (model.VM_rows(2, 1, TC, Eg, sigma, Jext), model.VM_rows(2, 1, TC[keep], Eg[keep], sigma[keep], Jext[keep])),
    ):
        assert not np.isfinite(rows.Iro[4]) or rows.Ptot[4] == 0.0
        # rows are solved independently (numpy's SIMD math differs in the last digit with the array position)
        np.testing.assert_allclose(rows.Ptot[keep], ref.Ptot, rtol=1e-9)


def test_rows_do_not_modify_the_device():
    model = Tandem3T()
    before = str(model)
    TC, Eg, sigma, Jext = _inputs(model)
    model.MPP_rows(TC, Eg, sigma, Jext)
    model.VM_rows(2, 1, TC, Eg, sigma, Jext)
    model.Isc3_rows(TC, Eg, sigma, Jext)
    assert str(model) == before
    assert model.top.JLC == Tandem3T().top.JLC and model.bot.JLC == Tandem3T().bot.JLC


def test_rows_reject_what_they_do_not_model():
    TC, Eg, sigma, Jext = _inputs(Tandem3T(), nrows=3)
    resistor = Tandem3T()
    resistor.top.set(pn=0)
    with pytest.raises(NotImplementedError, match="resistor-only"):
        resistor.MPP_rows(TC, Eg, sigma, Jext)
    with pytest.raises(ValueError, match="bot > 0 and top > 0"):
        Tandem3T().VM_rows(0, 1, TC, Eg, sigma, Jext)
    with pytest.raises(ValueError, match="3 rows"):
        Tandem3T().V3T_rows(IV3T(name="rows", shape=5), TC, Eg, sigma, Jext)
