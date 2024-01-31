"""Utilities for getting Earth-Sun vectors in a reference frame at a UTC epoch."""

# Standard library imports
from contextlib import closing
from datetime import datetime, timezone

# Third part imports
import numpy as np
from skyfield.api import load
from skyfield.sgp4lib import TEME


def get_earth_sun_vector_gcrs_at_epoch(epoch: datetime) -> np.ndarray:
    """Get the Earth-Sun position vector in GCRS at given epoch.

    Args:
        epoch (datetime): The epoch to use for getting the sun vector.

    Returns:
        earth_sun_vector_gcrs_at_epoch (np.ndarray): the position of the sun in GCRS frame at epoch.

    """
    # Argument checking
    if not isinstance(epoch, datetime):
        raise TypeError(f"arg 'epoch' must be of type datetime, not {type(epoch)}")

    ts = load.timescale()
    epoch = epoch.replace(tzinfo=timezone.utc)
    t = ts.from_datetime(epoch)
    bodies = load("de430_1850-2150.bsp")
    with closing(bodies):
        sun = bodies["sun"]
        earth = bodies["earth"]
        earth_sun_vector_gcrs_at_epoch = earth.at(t).observe(sun).position.km
        return np.array(earth_sun_vector_gcrs_at_epoch)


def get_earth_sun_vector_teme_at_epoch(epoch: datetime) -> np.ndarray:
    """Get the Earth-Sun position vector in TEME at given epoch.

    Args:
        epoch (datetime): The epoch to use for getting the sun vector.

    Returns:
        earth_sun_vector_teme_at_epoch (np.ndarray): the position of the sun in TEME frame at epoch.

    """
    # Argument checking
    if not isinstance(epoch, datetime):
        raise TypeError(f"arg 'epoch' must be of type datetime, not {type(epoch)}")

    ts = load.timescale()
    epoch = epoch.replace(tzinfo=timezone.utc)
    t = ts.from_datetime(epoch)
    bodies = load("de430_1850-2150.bsp")
    with closing(bodies):
        sun = bodies["sun"]
        earth = bodies["earth"]
        gcrs_vector = earth.at(t).observe(sun)
        teme_vector = gcrs_vector.frame_xyz(TEME).km
        return np.array(teme_vector)