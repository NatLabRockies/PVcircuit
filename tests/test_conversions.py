import numpy as np
import pandas as pd
import pytest
from scipy import constants

import pvcircuit as pvc
from pvcircuit import conversions as convert


def test_wavelength_to_photonenergy():
    np.testing.assert_almost_equal(convert.wavelength_to_photonenergy(1240.0), 1.0, decimal=4)


def test_photonenergy_to_wavelength():
    np.testing.assert_almost_equal(convert.wavelength_to_photonenergy(1.0), 1239.8419, decimal=4)


def test_normalize():
    df = pd.DataFrame([1, 2, 3, 4, 5])
    normalized_df = convert.normalize(df)
    np.testing.assert_almost_equal(normalized_df.to_numpy().flatten(), [0, 0.25, 0.5, 0.75, 1.0])


def test_TK():
    np.testing.assert_almost_equal(convert.TK(25.0), 298.15, decimal=2)


def test_Vth():
    k_q = constants.k / constants.e
    tc = 25.0
    expected_Vth = k_q * (tc + 273.15)
    np.testing.assert_almost_equal(convert.Vth(tc), expected_Vth, decimal=5)


# main
if __name__ == '__main__':
    pytest.main(['-v', __file__])