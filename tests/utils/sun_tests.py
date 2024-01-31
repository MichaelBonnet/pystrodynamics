"""Unit tests for Earth-Sun vector functions."""

# Standard library imports
from datetime import datetime

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
# from custom_logger import setup_logging
from pystrodynamics.utils.sun import get_earth_sun_vector_gcrs_at_epoch, get_earth_sun_vector_teme_at_epoch
from pystrodynamics.utils.math import angle_between_vectors

# get_earth_sun_vector_gcrs_at_epoch()
def test_get_earth_sun_vector_gcrs_at_epoch_with_freeflyer_data():

    # logging
    # logger = setup_logging(f"get_earth_sun_vector_gcrs_at_epoch_{datetime.now()}")

    test_data = pd.read_csv(
        "earth_sun_vector_test_data.csv"
    )

    test_count = 0

    # Validate the results
    for _, row in test_data.iterrows():
        test_count += 1
        
        # extract arg for function
        epoch = datetime.fromisoformat(row["Epoch_ISO8601_UTC"][:26])

        # get actual sun vector
        test_position_vector = get_earth_sun_vector_gcrs_at_epoch(epoch)

        # get desired sun vector
        true_position_vector = np.array(
            [
                row["Sun_Earth_MJ2000_X_km"],  # MJ2000 is supposedly accurate to GCRS within milliarcseconds
                row["Sun_Earth_MJ2000_Y_km"],
                row["Sun_Earth_MJ2000_Z_km"],
            ]
        )
        true_position_vector = -true_position_vector  # validation data has Sun-Earth, we want Earth-Sun

        angle_between = angle_between_vectors(test_position_vector, true_position_vector, "degrees")

        # logger.info(f"Test Case: {test_count}\tCalculated Vector: {test_position_vector}\tTrue Vector: {true_position_vector}\tAngle Diff (degs): {angle_between}\tAngle Diff (arcsec): {angle_between * 3600}")

        # accurate to 1/10th of 1%
        np.testing.assert_allclose(test_position_vector, true_position_vector, rtol=0.001)
        assert np.abs(angle_between) < 0.01


# get_earth_sun_vector_teme_at_epoch()
def test_get_earth_sun_vector_teme_at_epoch_with_freeflyer_data():

    # logging
    # logger = setup_logging(f"get_earth_sun_vector_teme_at_epoch_{datetime.now()}")

    test_data = pd.read_csv(
        "earth_sun_vector_test_data.csv"
    )

    test_count = 0

    # Validate the results
    for _, row in test_data.iterrows():
        test_count += 1

        # extract arg for function
        epoch = datetime.fromisoformat(row["Epoch_ISO8601_UTC"][:26])

        # get actual sun vector
        test_position_vector = get_earth_sun_vector_teme_at_epoch(epoch)

        # get desired sun vector
        true_position_vector = np.array(
            [
                row["Sun_Earth_TEME_X_km"],
                row["Sun_Earth_TEME_Y_km"],
                row["Sun_Earth_TEME_Z_km"],
            ]
        )
        true_position_vector = -true_position_vector  # validation data has Sun-Earth, we want Earth-Sun

        angle_between = angle_between_vectors(test_position_vector, true_position_vector, "degrees")

        # logger.info(f"Test Case: {test_count}\tCalculated Vector: {test_position_vector}\tTrue Vector: {true_position_vector}\tAngle Diff (degs): {angle_between}\tAngle Diff (arcsec): {angle_between * 3600}")

        # accurate to one-tenth of 1% on each element
        np.testing.assert_allclose(test_position_vector, true_position_vector, rtol=0.001)
        # accurate to about ~0.00578 degrees, or about 21 arcseconds
        assert np.abs(angle_between) < 0.01
