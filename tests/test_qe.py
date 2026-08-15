import matplotlib
matplotlib.use('Agg')  # Add this line at the top, before importing pyplot
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

import pvcircuit as pvc
from pvcircuit import EQE
from pvcircuit.qe import wvl, AM15G


def get_measured_eqe():
    path = pvc.notebook_datapath
    eqe_file = "MM927Bn5CEQE.csv"
    data = pd.read_csv(path.joinpath(eqe_file), index_col=0)
    eqe = pvc.EQE(data.index, data, "TestEQE")
    return eqe


@pytest.fixture
def ideal_eqe():
    wvl = np.arange(280, 4000)
    return pvc.EQE(wvl, np.ones_like(wvl))


@pytest.fixture
def example_eqe():
    path = pvc.notebook_datapath
    eqe_file = "MM927Bn5CEQE.csv"
    data = pd.read_csv(path.joinpath(eqe_file), index_col=0)
    eqe = pvc.EQE(data.index, data, "TestEQE")
    return eqe


@pytest.fixture
def example_eqeT():
    path = pvc.notebook_datapath
    eqe_file = "MP846n8.csv"
    data = pd.read_csv(path.joinpath(eqe_file), index_col=0)
    temperatures = data.columns.to_series().str.findall(r"(\d+)C").explode().dropna().astype(int)
    eqeT = pvc.qe.EQET(data.index, data, temperatures, "TestEQE")
    return eqeT


def test_eqe(ideal_eqe):
    ideal_eqe.add_spectra()
    np.testing.assert_almost_equal(68.98763776603343, ideal_eqe.Jint())


def test_Jint():
    waves = np.arange(300, 1200)
    eq = np.ones_like(waves)
    eqe = EQE(waves, eq)

    eqe.add_spectra(wvl, AM15G.T)
    # Jint() always returns the (n_eqe, n_spectra) matrix
    test_res = np.array([[46.42154037]])
    np.testing.assert_array_almost_equal(test_res, eqe.Jint())
    # pairwise: one spectrum per EQE column -> 1-D
    np.testing.assert_array_almost_equal(np.array([46.42154037]), eqe.Jint(pairwise=True))

    eqe.add_spectra(wvl, np.tile(AM15G.T, [5, 1]).T)
    test_res = np.array([[46.42154037, 46.42154037, 46.42154037, 46.42154037, 46.42154037]])
    np.testing.assert_array_almost_equal(test_res, eqe.Jint())
    with pytest.raises(ValueError, match="needs one spectrum per EQE column"):
        eqe.Jint(pairwise=True)

    eqe.add_eqe(waves, eq * 0.5)
    test_res = np.vstack([test_res, test_res / 2])
    np.testing.assert_array_almost_equal(test_res, eqe.Jint())

    # deprecated alias still works
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_almost_equal(test_res, eqe.Jint(enforce_all_combinations=True))


def test_Jint_no_pairwise_by_coincidence():
    """2 junctions x 2 spectra must give the 2x2 matrix, not [J_top(spec1), J_bot(spec2)]."""
    lam = np.linspace(300, 1200, 901)
    eqe = EQE(lam, np.column_stack([np.where(lam < 700, 0.9, 0.0), np.where(lam >= 700, 0.9, 0.0)]), name="t", sjuncs=["top", "bot"])
    eqe.add_spectra(lam, np.column_stack([np.ones_like(lam), 0.5 * np.ones_like(lam)]))
    j = eqe.Jint()
    assert j.shape == (2, 2)
    np.testing.assert_allclose(j[:, 1], 0.5 * j[:, 0])
    # explicit pairing gives the diagonal
    np.testing.assert_allclose(eqe.Jint(pairwise=True), np.diag(j))


def test_ordinal():
    np.testing.assert_string_equal(pvc.qe.ordinal(1), "1st")
    np.testing.assert_string_equal(pvc.qe.ordinal(2), "2nd")
    np.testing.assert_string_equal(pvc.qe.ordinal(3), "3rd")
    np.testing.assert_string_equal(pvc.qe.ordinal(4), "4th")
    np.testing.assert_string_equal(pvc.qe.ordinal(11), "11th")
    np.testing.assert_string_equal(pvc.qe.ordinal(21), "21st")


def test_eq_solve_Eg():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    result = pvc.qe._eq_solve_Eg(1.0, x, y)
    np.testing.assert_almost_equal(result, 1.1)


def test_gaussian():
    x = np.array([1, 2, 3])
    result = pvc.qe._gaussian(x, 1, 2, 1)
    np.testing.assert_almost_equal(result, [0.60653066, 1.0, 0.60653066])


def test_PintMD():
    Pspec = "global"
    np.testing.assert_almost_equal(pvc.qe.PintMD(Pspec), 1000.4707036093448)


def test_JintMD(example_eqe):
    Pspec = "global"
    result = pvc.qe.JintMD(example_eqe.eqe, example_eqe.wavelength.flatten(), Pspec)
    np.testing.assert_array_almost_equal(result, np.array([[13.3176649, 12.80026233, 12.15024507, 11.51752523]]))


def test_JdbMD(example_eqe):
    jdb, bandgaps = pvc.qe.JdbMD(example_eqe.eqe, example_eqe.wavelength.flatten(), 25)
    jdb, bandgaps = pvc.qe.JdbMD(example_eqe.eqe, example_eqe.wavelength.flatten(), 25, bplot=True)
    np.testing.assert_array_almost_equal(jdb, np.array([1.61112541e-28, 1.22172821e-21, 6.33343919e-16, 6.36701070e-11]))
    np.testing.assert_array_almost_equal(bandgaps, np.array([1.83036399, 1.41017015, 1.0576233, 0.74415209]))


def test_JdbFromEg():
    TC = 300
    Eg = 1.12
    np.testing.assert_almost_equal(pvc.qe.JdbFromEg(TC, Eg), 1.5193186044138e-07)


def test_EgFromJdb():
    TC = 300
    Jdb = 1e-11
    np.testing.assert_almost_equal(pvc.qe.EgFromJdb(TC, Jdb), 1.631346190417939)


def test_ensure_numpy_2drow():
    array = [1, 2, 3]
    result = pvc.qe.ensure_numpy_2drow(array)
    np.testing.assert_array_equal(result.shape, np.array([1, 3]))


def test_ensure_numpy_2dcol():
    array = [1, 2, 3, 4]
    result = pvc.qe.ensure_numpy_2dcol(array)
    np.testing.assert_array_equal(result.shape, np.array([4, 1]))


def test_plot(example_eqe):
    lax, rax = example_eqe.plot()
    assert isinstance(lax, plt.Axes)
    assert isinstance(rax, plt.Axes)


def test_plotsr(example_eqe):
    lax, rax = example_eqe.plot_sr()
    assert isinstance(lax, plt.Figure)
    assert isinstance(rax, plt.Axes)


def test_EQE(example_eqe):

    np.testing.assert_equal(example_eqe.njuncs, 4)
    np.testing.assert_string_equal(example_eqe.name, "TestEQE")

    example_eqe.add_spectra()
    np.testing.assert_array_almost_equal(example_eqe.Jint(), np.array([[13.36347958], [12.72695153], [12.10891819], [11.58554508]]))

    bandgaps, sigmas = example_eqe.calc_Eg_Rau()
    bandgaps, sigmas = example_eqe.calc_Eg_Rau(plot_fits=True)  # for test coverage
    np.testing.assert_array_almost_equal(bandgaps, np.array([1.8280609, 1.40411546, 1.05180891, 0.74064824]))
    np.testing.assert_array_almost_equal(sigmas, np.array([0.03637069, 0.01110312, 0.01503357, 0.01186747]))

    Jdb, Egnew = example_eqe.Jdb(320)
    np.testing.assert_array_almost_equal(Jdb, np.array([6.374705e-13, 1.401277e-09, 7.752416e-07, 1.920538e-04]))
    np.testing.assert_array_almost_equal(Egnew, np.array([1.84294, 1.424075, 1.073562, 0.758273]))

    waves = np.arange(300, 1200)
    data = np.ones_like(waves) * 0.9
    example_eqe.add_eqe(waves, data, 20)
    np.testing.assert_equal(example_eqe.njuncs, 5)
    np.testing.assert_array_equal(example_eqe.eqe[example_eqe.wavelength.flatten() > waves.max(), -1], 0)


def test_EQET(example_eqe):
    example_eqe = pvc.qe.EQET(example_eqe.wavelength, example_eqe.eqe, np.repeat(25, example_eqe.njuncs), "TestEQET")
    np.testing.assert_equal(example_eqe.njuncs, 4)
    np.testing.assert_string_equal(example_eqe.name, "TestEQET")

    example_eqe.add_spectra()
    np.testing.assert_array_almost_equal(example_eqe.Jint(), np.array([[13.36347958], [12.72695153], [12.10891819], [11.58554508]]))

    bandgaps, sigmas = example_eqe.calc_Eg_Rau()
    np.testing.assert_array_almost_equal(bandgaps, np.array([1.8280609, 1.40411546, 1.05180891, 0.74064824]))
    np.testing.assert_array_almost_equal(sigmas, np.array([0.03637069, 0.01110312, 0.01503357, 0.01186747]))

    Jdb, Egnew = example_eqe.Jdb(320)
    np.testing.assert_array_almost_equal(Jdb, np.array([6.374705e-13, 1.401277e-09, 7.752416e-07, 1.920538e-04]))
    np.testing.assert_array_almost_equal(Egnew, np.array([1.84294, 1.424075, 1.073562, 0.758273]))

    waves = np.arange(300, 1200)
    data = np.ones_like(waves) * 0.9
    example_eqe.add_eqe(waves, data, 20)
    np.testing.assert_equal(example_eqe.njuncs, 5)
    # note that that this is the first eqe in the array as the adding sorts the eqe array by temperature 20 < 25...
    np.testing.assert_array_equal(example_eqe.eqe[example_eqe.wavelength.flatten() > waves.max(), 0], 0)


def test_get_current_for_temperature(example_eqeT):

    with pytest.raises(ValueError, match="Load spectral information first."):
        result = example_eqeT.get_current_for_temperature(25)

    example_eqeT.add_spectra()  # loads AM15G
    currents = example_eqeT.Jint()
    result = example_eqeT.get_current_for_temperature([25])
    np.testing.assert_almost_equal(result, currents[0], decimal=2)

    example_eqeT.add_spectra(example_eqeT.wavelength.flatten(), np.repeat(example_eqeT.spectra, len(example_eqeT.temperature), axis=1))  # interpoalte dimension must be lower than the original
    result = example_eqeT.get_current_for_temperature(example_eqeT.temperature)
    np.testing.assert_array_almost_equal(result, currents.flatten(), decimal=1)

    result = example_eqeT.get_current_for_temperature(example_eqeT.temperature, degrees=[1, 2, 3, 4, 5])  # degree 5 should still be the best
    np.testing.assert_array_almost_equal(result, currents.flatten(), decimal=1)


def test_eqeTplot(example_eqeT):
    lax, rax = example_eqeT.plot()
    assert isinstance(lax, plt.Axes)
    assert isinstance(rax, plt.Axes)


def test_eqeTplotsr(example_eqeT):
    lax, rax = example_eqeT.plot_sr()
    assert isinstance(lax, plt.Figure)
    assert isinstance(rax, plt.Axes)


def test_TemperatureModel():
    x = np.array([25, 50, 75, 100])
    y = np.array([1.0, 0.9, 0.8, 0.7])
    model = pvc.qe.TemperatureModel.fit(x, y, model_types=[pvc.qe.ModelType.LINEAR, pvc.qe.ModelType.POLY2, pvc.qe.ModelType.POLY3, pvc.qe.ModelType.POLY4, pvc.qe.ModelType.POLY5])
    model = pvc.qe.TemperatureModel.fit(x, y, model_types=[pvc.qe.ModelType.LINEAR, pvc.qe.ModelType.POLY2, pvc.qe.ModelType.POLY3, pvc.qe.ModelType.POLY4, pvc.qe.ModelType.POLY5], plot="all")
    model = pvc.qe.TemperatureModel.fit(x, y, model_types=[pvc.qe.ModelType.LINEAR, pvc.qe.ModelType.POLY2, pvc.qe.ModelType.POLY3, pvc.qe.ModelType.POLY4, pvc.qe.ModelType.POLY5], plot="best")
    # should be linear
    assert model.model_type == pvc.qe.ModelType.LINEAR
    np.testing.assert_almost_equal(model.apply(50, 1.0), y[1])


def test_LCcorr(example_eqe):
    """LCcorr applies the Steiner 2013 luminescent-coupling correction.
    With all etas = 0 the corrected EQE equals the raw EQE.  Setting
    eta[1,0] > 0 leaves the first-junction's corrected EQE untouched
    (always raw[:,0]) and changes the second junction's corrected EQE
    per the Steiner formula corrEQE[:,1] = raw[:,1]*(1+eta) - raw[:,0]*eta.
    """
    # Default: all etas zero -> corrEQE == raw EQE
    example_eqe.LCcorr()
    np.testing.assert_array_almost_equal(example_eqe.corrEQE, example_eqe.eqe)

    # First-junction's corrected EQE is the raw value regardless of any etas
    raw_top_before = example_eqe.eqe[:, 0].copy()

    # Set eta[1, 0] = 0.1 -- couples top->bot
    example_eqe.LCcorr(junc=1, dist=0, val=0.1)
    np.testing.assert_array_almost_equal(example_eqe.corrEQE[:, 0], raw_top_before)

    # Bottom-junction's corrected EQE follows the Steiner formula
    eta = 0.1
    expected_bot = example_eqe.eqe[:, 1] * (1.0 + eta) - example_eqe.eqe[:, 0] * eta
    np.testing.assert_array_almost_equal(example_eqe.corrEQE[:, 1], expected_bot)

    # Setting eta back to 0 reverts the bot correction to the raw EQE
    example_eqe.LCcorr(junc=1, dist=0, val=0.0)
    np.testing.assert_array_almost_equal(example_eqe.corrEQE[:, 1], example_eqe.eqe[:, 1])


def test_EQET_extrapolation_beyond_data(example_eqeT):
    """get_eqe_at_temperature uses scipy interp1d with fill_value='extrapolate',
    and clips negative results to zero.  Extrapolating well beyond the
    measured temperature range must therefore return a valid EQET object
    (no crash, no negatives)."""
    t_min = float(example_eqeT.temperature.min())
    t_max = float(example_eqeT.temperature.max())

    # Inside the measured range
    inside = example_eqeT.get_eqe_at_temperature(t_min + 1.0)
    assert isinstance(inside, pvc.qe.EQET)
    assert np.all(inside.eqe >= 0)

    # Well above the measured range: extrapolation, no crash, clipped to [0, max]
    above = example_eqeT.get_eqe_at_temperature(t_max + 50.0)
    assert isinstance(above, pvc.qe.EQET)
    assert above.njuncs == 1
    np.testing.assert_array_equal(above.wavelength, example_eqeT.wavelength)
    assert np.all(above.eqe >= 0)
    assert np.all(above.eqe <= example_eqeT.eqe.max() + 1e-9)

    # Well below the measured range: same protections
    below = example_eqeT.get_eqe_at_temperature(t_min - 50.0)
    assert isinstance(below, pvc.qe.EQET)
    assert np.all(below.eqe >= 0)
    assert np.all(below.eqe <= example_eqeT.eqe.max() + 1e-9)


def test_EQE_add_eqe(example_eqe):
    """add_eqe appends one more junction column at potentially-different
    wavelengths.  njuncs increments by one, eqe gains one column, and the
    sjuncs label list grows.  New wavelengths outside the original range
    are zero-padded by the interp1d fill_value=0."""
    n_before = example_eqe.njuncs
    sjuncs_before = list(example_eqe.sjuncs)

    waves = np.arange(300, 1200)
    eqe_extra = np.ones_like(waves) * 0.5
    example_eqe.add_eqe(waves, eqe_extra, sjuncs="extra_junc")

    assert example_eqe.njuncs == n_before + 1
    assert example_eqe.eqe.shape[1] == n_before + 1
    assert example_eqe.sjuncs == sjuncs_before + ["extra_junc"]

    # New column is the last one -- outside [300, 1200] it must be zero-padded
    new_col = example_eqe.eqe[:, -1]
    outside_mask = example_eqe.wavelength.flatten() > waves.max()
    if outside_mask.any():
        np.testing.assert_array_equal(new_col[outside_mask], 0)


if __name__ == "__main__":

    example_eqe = get_measured_eqe()

    example_eqe.add_spectra()
    example_eqe.plot()
    plt.show()
