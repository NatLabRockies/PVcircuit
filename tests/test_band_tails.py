# -*- coding: utf-8 -*-
"""Synthetic ground-truth tests for the Katahara theta band-tail extension.

These tests cover the generalization of `pvcircuit.junction.Jdb` from the
historical Mattheis-Rau-Werner Gaussian (theta=2) to arbitrary tail-shape
exponents, and the `Junction.theta` attribute round-trip through copy /
pickle / set().
"""
import copy
import pickle

import numpy as np
import pytest
from scipy.integrate import quad

import pvcircuit as pvc
from pvcircuit.conversions import DB_PREFIX, TK, Vth
from pvcircuit.junction import Jdb, Junction


# ---------------------------------------------------------------------------
# Jdb() generalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [1.0, 1.25, 1.5, 2.0])
def test_sigma_zero_invariant_across_theta(theta):
    """sigma=0 collapses to the SQ step result independent of theta.

    Guards the early-return branch in `Jdb()`. Compared against the
    hard-coded baseline value in test_junction.py.
    """
    TC = 25.0
    Eg = 1.12
    expected_sq = 6.249646867228706e-17
    np.testing.assert_allclose(Jdb(TC, Eg, sigma=0.0, theta=theta), expected_sq, rtol=1e-12)


def test_theta_2_unchanged_under_nonzero_sigma():
    """Regression guard: the theta=2 branch reproduces the Mattheis-Rau-Werner
    closed form exactly for a non-trivial sigma. If somebody refactors the
    dispatch logic, this test catches accidental drift.
    """
    TC = 25.0
    Eg = 1.42
    sigma = 20e-3
    Vthlocal = Vth(TC)
    EgkT = Eg / Vthlocal
    # Recompute the historical Mattheis-Rau-Werner expression inline.
    expected = (
        DB_PREFIX
        * TK(TC) ** 3.0
        * (
            EgkT * EgkT
            + 2.0 * EgkT
            + 2.0
            - 2.0 * sigma**2 * Eg / Vthlocal**3
            - sigma**2 / Vthlocal**2
            + sigma**4 / Vthlocal**4
        )
        * np.exp(-EgkT + sigma**2 / (2.0 * Vthlocal**2))
    )
    np.testing.assert_allclose(Jdb(TC, Eg, sigma=sigma, theta=2.0), expected, rtol=1e-12)


@pytest.mark.parametrize("theta", [1.0, 1.25, 1.5])
@pytest.mark.parametrize("sigma", [2e-3, 20e-3])
def test_urbach_numerical_matches_quad_reference(theta, sigma):
    """The numerical path matches an independent high-precision reference
    with epsabs=0.0 (and the exact analytic closed form at theta=1),
    including narrow tails (sigma=2 meV) where default epsabs=1.49e-8 fails.
    """
    TC = 25.0
    Eg = 1.5
    Vthlocal = Vth(TC)
    EgkT = Eg / Vthlocal
    sq_bracket = EgkT * EgkT + 2.0 * EgkT + 2.0

    if theta == 1.0:
        a = Vthlocal / sigma - 1.0
        below_eg_ref = np.exp(-EgkT) * (
            EgkT**2 / a - 2.0 * EgkT / a**2 + 2.0 / a**3 - 2.0 * np.exp(-a * EgkT) / a**3
        )
    else:
        # Factor out exp(-EgkT) in u = EgkT - x so integrand is O(1) near the band edge
        def reference_integrand(u):
            x = EgkT - u
            ratio = u * Vthlocal / sigma
            return np.exp(-(ratio ** theta) + u) * x * x

        below_eg_scaled, _ = quad(reference_integrand, 0.0, EgkT, limit=500, epsrel=1e-13, epsabs=0.0)
        below_eg_ref = np.exp(-EgkT) * below_eg_scaled

    expected = DB_PREFIX * TK(TC) ** 3.0 * (np.exp(-EgkT) * sq_bracket + below_eg_ref)

    np.testing.assert_allclose(Jdb(TC, Eg, sigma=sigma, theta=theta), expected, rtol=1e-10)


@pytest.mark.parametrize("theta", [1.0, 1.25, 1.5])
def test_urbach_zero_sigma_limit(theta):
    """As sigma -> 0 the generalized-Urbach result approaches the SQ value.

    For sigma below the early-return threshold (sigma = 0 exactly) the
    branch returns SQ; we check that arbitrarily small but non-zero sigma
    is close to SQ to within the integrand's precision. This guards
    against e.g. a sign error in the integrand that would diverge as
    sigma -> 0 rather than vanishing.
    """
    TC = 25.0
    Eg = 1.42
    sq = Jdb(TC, Eg, sigma=0.0)
    # 1e-6 eV is far below any physical Urbach energy and well below Vth (~26 meV).
    tailed = Jdb(TC, Eg, sigma=1e-6, theta=theta)
    np.testing.assert_allclose(tailed, sq, rtol=1e-6)


def test_jdb_monotone_in_sigma_for_theta_1():
    """Larger Urbach energy -> more sub-bandgap absorption -> larger Jdb.

    Tests the physical monotonicity of the generalized-Urbach branch with
    theta=1 (true Urbach, the relevant case for perovskite / a-Si).
    """
    TC = 25.0
    Eg = 1.5
    sigmas = [5e-3, 10e-3, 20e-3, 30e-3]
    j0_values = [Jdb(TC, Eg, sigma=s, theta=1.0) for s in sigmas]
    # Strictly increasing
    for prev, curr in zip(j0_values, j0_values[1:]):
        assert curr > prev, f"Jdb should grow with sigma but got {j0_values}"


def test_warning_for_extreme_sigma():
    """sigma > 3*Vth triggers a UserWarning in the generalized-Urbach path.

    Guards against silently accepting nonsense inputs. The warning text
    mentions 'unphysical' so callers can grep / filter it.
    """
    TC = 25.0
    Vthlocal = Vth(TC)  # ~25.7 meV at 25 C
    big_sigma = 4.0 * Vthlocal  # ~103 meV, comfortably above the 3*Vth gate
    with pytest.warns(UserWarning, match="unphysical"):
        Jdb(TC, 1.5, sigma=big_sigma, theta=1.0)


# ---------------------------------------------------------------------------
# Junction.theta attribute
# ---------------------------------------------------------------------------


def test_junction_default_theta_is_2():
    """Default Junction picks theta=2.0 to preserve historical behaviour."""
    j = Junction()
    assert float(j.theta) == 2.0


def test_junction_set_theta_changes_jdb():
    """Mutating theta via set() changes Jdb when sigma > 0.

    With sigma=0 both branches collapse to SQ, so we use a non-trivial sigma
    to exercise the branch dispatch.
    """
    j = Junction(Eg=1.5, sigma=20e-3, n=[1.0], J0ratio=[1.0])
    j.set(theta=2.0)
    j0_gaussian = float(j.Jdb)
    j.set(theta=1.0)
    j0_urbach = float(j.Jdb)
    assert j0_gaussian != j0_urbach


def test_junction_str_omits_theta_when_default():
    """__str__ keeps the historical line format for theta=2.0 devices.

    Existing baseline files (tests/test_files/Junction.txt, Tandem3T_str.txt)
    were generated before theta existed; this test guards that they stay
    valid for the default-theta case.
    """
    j = Junction()
    assert "theta" not in str(j)


def test_junction_str_shows_theta_when_nondefault():
    """__str__ appends the theta value when it differs from the default.

    Visible to the user but does not affect the default-case baseline.
    """
    j = Junction()
    j.set(theta=1.0)
    assert "theta" in str(j)


def test_junction_theta_survives_copy():
    """Junction.copy() preserves the theta attribute."""
    j = Junction(Eg=1.5, sigma=20e-3)
    j.set(theta=1.25)
    j2 = j.copy()
    assert float(j2.theta) == 1.25
    # Mutating the copy must not leak back to the original.
    j2.set(theta=2.0)
    assert float(j.theta) == 1.25


def test_junction_theta_survives_pickle():
    """Junction.theta round-trips through pickle (matches the
    test_serialization.py pattern for other attrs).
    """
    j = Junction(Eg=1.5, sigma=20e-3)
    j.set(theta=1.5)
    blob = pickle.dumps(j)
    j2 = pickle.loads(blob)
    assert float(j2.theta) == 1.5
    np.testing.assert_allclose(float(j2.Jdb), float(j.Jdb), rtol=1e-12)


def test_old_pickle_without_theta_attribute_still_works(monkeypatch):
    """Old-format Junction missing the `theta` attribute defaults to 2.0.

    Simulates loading a pickle written before Phase 2 by deleting the attr
    after construction. The Junction.Jdb property uses getattr(default=2.0)
    so this must keep working.
    """
    j = Junction(Eg=1.5, sigma=20e-3)
    # Simulate the old pickle layout.
    del j.__dict__["theta"]
    # Should not raise, should give the Mattheis-Rau-Werner value.
    expected = Jdb(j.TC, j.Eg, sigma=j.sigma, theta=2.0)
    np.testing.assert_allclose(float(j.Jdb), expected, rtol=1e-12)
