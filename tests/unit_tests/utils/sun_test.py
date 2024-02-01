"""Unit tests for Earth-Sun vector functions.

This module contains unit tests for functions related to Earth-Sun vector calculations.
These tests validate the accuracy of the Earth-Sun vector calculations in different frames
and provide test cases with expected results.
"""

# Standard library imports
from datetime import datetime

# Third party imports
import numpy as np
import pandas as pd
import pytest

# Local application imports
# from custom_logger import setup_logging
from pystrodynamics.utils.sun import get_earth_sun_vector_gcrs_at_epoch, get_earth_sun_vector_teme_at_epoch
from pystrodynamics.utils.math import angle_between_vectors

@pytest.fixture
def test_data():
    """Fixture to load test data from a CSV file."""
    yield pd.read_csv("tests/data/earth_sun_vector_test_data.csv")


def test_get_earth_sun_vector_gcrs_at_epoch_with_freeflyer_data(test_data):
    """Test function for get_earth_sun_vector_gcrs_at_epoch with freeflyer data.

    Args:
        test_data (pd.DataFrame): Test data containing epochs and expected results.

    This test function iterates through the provided test data and validates the accuracy
    of the calculated Earth-Sun vector in the GCRS frame at the given epochs.
    """

    # logging
    # logger = setup_logging(f"get_earth_sun_vector_gcrs_at_epoch_{datetime.now()}")

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

        # accurate to within milliarcseconds
        assert np.abs(angle_between) < 0.00001


def test_get_earth_sun_vector_teme_at_epoch_with_freeflyer_data(test_data):
    """Test function for get_earth_sun_vector_teme_at_epoch with freeflyer data.

    Args:
        test_data (pd.DataFrame): Test data containing epochs and expected results.

    This test function iterates through the provided test data and validates the accuracy
    of the calculated Earth-Sun vector in the TEME frame at the given epochs.
    """

    # logging
    # logger = setup_logging(f"get_earth_sun_vector_teme_at_epoch_{datetime.now()}")

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

        # accurate to within milliarcseconds
        assert np.abs(angle_between) < 0.00001
