import multiprocessing as mp
import pickle
from pathlib import Path
import tempfile
import numpy as np
import pytest
import pvcircuit as pvc
from pvcircuit import EQE, Junction, Multi2T, Tandem3T


def worker(obj):
    """Helper function for multiprocessing."""
    return pickle.dumps(obj)


def get_numeric_attributes(obj):
    """
    Extract all numeric attributes (float, int, numpy.ndarray, pandas objects) from an object.

    :param obj: The object to inspect.
    :return: A dictionary of numeric attributes and their values.
    """
    numeric_attrs = {}
    for attr in dir(obj):
        if not attr.startswith("_"):  # Skip private/protected attributes
            value = getattr(obj, attr, None)
            if isinstance(value, (int, float, np.ndarray)):
                numeric_attrs[attr] = value
            try:
                # Check for pandas Series or DataFrame if pandas is used
                import pandas as pd
                if isinstance(value, (pd.Series, pd.DataFrame)):
                    numeric_attrs[attr] = value.values
            except ImportError:
                pass
    return numeric_attrs


def compare_numeric_attributes(obj1, obj2):
    """
    Compare numeric attributes between two objects.

    :param obj1: First object.
    :param obj2: Second object.
    """
    attrs1 = get_numeric_attributes(obj1)
    attrs2 = get_numeric_attributes(obj2)

    # Compare keys
    np.testing.assert_array_equal(sorted(attrs1.keys()), sorted(attrs2.keys()),
                                  err_msg="Attribute keys mismatch")

    # Compare values
    for key in attrs1:
        val1, val2 = attrs1[key], attrs2[key]
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            np.testing.assert_array_equal(val1, val2, err_msg=f"Mismatch in attribute '{key}'")
        else:
            np.testing.assert_almost_equal(val1, val2, err_msg=f"Mismatch in attribute '{key}'")


def run_serialization_test(obj):
    """
    Generalized serialization test for different objects.

    :param obj: The object to serialize and test.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
        temp_file_name = temp_file.name

    try:
        # Serialize to file
        with open(temp_file_name, "wb") as f:
            pickle.dump(obj, f)

        # Deserialize from file
        with open(temp_file_name, "rb") as f:
            obj_from_file = pickle.load(f)

        # Compare numeric attributes
        compare_numeric_attributes(obj, obj_from_file)

        # Test with multiprocessing
        with mp.Pool(1) as pool:
            result = pool.apply(worker, (obj,))
            obj_from_mp = pickle.loads(result)

        # Compare numeric attributes
        compare_numeric_attributes(obj, obj_from_mp)
    finally:
        Path(temp_file_name).unlink()


def test_IV3T():
    T3 = Tandem3T()
    iv3t = T3.VM(2, 1)
    run_serialization_test(iv3t)


def test_EQE():
    waves = np.arange(280, 4000)
    data = np.ones_like(waves) * 0.8
    eqe = EQE(waves, data)
    run_serialization_test(eqe)


def test_Junction():
    junction = Junction(name="test_junction")
    run_serialization_test(junction)


def test_Multi2T():
    multi2t = Multi2T()
    run_serialization_test(multi2t)


def test_Tandem3T():
    tandem3t = Tandem3T()
    run_serialization_test(tandem3t)


def test_EQET():
    """EQET round-trip: same data shape, temperature axis preserved."""
    from pvcircuit.qe import EQET

    waves = np.arange(300, 1200, 10)
    temperatures = np.array([25, 50, 75])
    # eqe shape (N_wvl, N_temp)
    eqe_data = np.tile(np.linspace(0.1, 0.9, len(waves))[:, np.newaxis], (1, len(temperatures)))
    eqet = EQET(waves, eqe_data, temperatures)
    run_serialization_test(eqet)


def test_Meteo():
    """Meteo round-trip: small synthetic time-series, no external file needed."""
    import pandas as pd
    from pvcircuit.EY import Meteo

    n = 5
    waves = np.arange(300, 1200, 50)
    idx = pd.date_range("2024-06-01 09:00", periods=n, freq="1h", tz="UTC")
    # spectra: (n, N_wvl) DataFrame indexed by datetime
    spectra = pd.DataFrame(
        np.tile(np.linspace(0.5, 1.5, len(waves)), (n, 1)),
        index=idx,
        columns=waves,
    )
    temp = pd.Series(np.linspace(15.0, 25.0, n), index=idx)
    wind = pd.Series(np.linspace(1.0, 3.0, n), index=idx)

    meteo = Meteo(waves, spectra, temp, wind, idx)
    run_serialization_test(meteo)
