from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass

import numpy as np
from sgp4.api import Satrec, WGS72
from sgp4.io import twoline2rv, rv2coe
from sgp4.earth_gravity import wgs72
from skyfield.api import EarthSatellite, load


@dataclass
class ClassicalOrbitalElements:
    """Dataclass for the classical orbital elements returned by the rv2coe function.
    
    Attributes:
        semilatus_rectum (float):
            shorthand: p
             Units: km
            Range: 0 to np.inf
        semimajor_axis (float):
            shorthand: a)
            Units: km
        eccentricity (float):
            shorthand: ecc)
            Units: N/A
            Range: 0 to np.inf
        inclination (float):
            shorthand: incl)
            Units: rad
            Range: 0.0  to pi rad
        longitude_of_ascending_node (float):
            shorthand: omega)
            Units: rad
            Range: 0.0  to 2pi rad
        argument_of_perigee (float):
            shorthand: argp)
            Units: rad
            Range: 0.0  to 2pi rad
        true_anomaly (float):
            shorthand: nu)
            Units: rad
            Range: 0.0  to 2pi rad
        mean_anomaly (float):
            shorthand: m)
            Units: rad
            Range: 0.0  to 2pi rad
        argument_of_latitude (float):
            shorthand: arglat) (ci)
            Units: rad
            Range: 0.0  to 2pi rad
        true_longitude (float):
            shorthand: truelon) (ce)
            Units: rad
            Range: 0.0  to 2pi rad
        longitude_of_periapsis (float):
            shorthand: lonper) (ee)
            Units: rad 
         Range: 0.0  to 2pi rad
    """
    semilatus_rectum: float
    semimajor_axis: float
    eccentricity: float
    inclination: float
    longitude_of_ascending_node: float
    argument_of_perigee: float
    true_anomaly: float
    mean_anomaly: float
    argument_of_latitude: float
    true_longtitude: float
    longitube_of_periapsis: float

def tle_and_epoch_to_state_vectors(
    tle_line1: str, tle_line2: str, epoch: datetime, reference_frame: Optional[str] = "TEME"
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the position and velocity vectors of a spacecraft in TEME frame given its TLE and an epoch.

    Args:
        tle_line1 (str): the first line of the object's TLE
        tle_line2 (str): the second line of the object's TLE
        epoch (datetime.datetime): the epoch for which to calculate object position and velocity

    Returns:
        position_vector (np.ndarray): the position vector of the object in the desired frame at given epoch
        velocity_vector (np.ndarray): the velocity vector of the object in the desired frame at given epoch

    """
    epoch = epoch.replace(tzinfo=timezone.utc)
    if reference_frame == "TEME":
        # Parse the TLE data
        satellite = Satrec.twoline2rv(tle_line1, tle_line2, WGS72)

        # Calculate the state vector at the specified time
        position_tuple, velocity_tuple = satellite.propagate(
            epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, (epoch.second + (epoch.microsecond / 1000000))
        )

        position_vector_teme = np.array(position_tuple)
        velocity_vector_teme = np.array(velocity_tuple)

        return position_vector_teme, velocity_vector_teme
    elif reference_frame == "GCRS":
        ts = load.timescale()
        satellite = EarthSatellite(tle_line1, tle_line2, ts)

        geocentric = satellite.at(ts.from_datetime(epoch))
        position_vector_gcrs = geocentric.position.km
        velocity_vector_gcrs = geocentric.velocity.km_s

        return position_vector_gcrs, velocity_vector_gcrs

def state_vectors_to_coe(position_vector: np.ndarray, velocity_vector: np.ndarray) -> ClassicalOrbitalElements:
    return ClassicalOrbitalElements(rv2coe(position_vector, velocity_vector, wgs72.mu))
