# Standard library imports
from datetime import datetime
from typing import Optional

# Third party imports
import numpy as np
from scipy.spatial.transform import Rotation as R

# Local imports
from pystrodynamics.simulation_objects.orbital_object import OrbitalObject
from pystrodynamics.utils.sun import get_earth_sun_vector_gcrs_at_epoch, get_earth_sun_vector_teme_at_epoch
from pystrodynamics.simulation_objects.spacecraft_modules.basic_sensor import BasicSensor
from pystrodynamics.utils.rotations import gcrs_to_lvlh_rotation


class Spacecraft(OrbitalObject):
    """
    Represents a spacecraft for simulation purposes, including propagation based on TLE (Two-Line Elements) data
    and the ability to compute rotations from body to various reference frames.
    
    Inherits from OrbitalObject to utilize orbital mechanics and state propagation functionalities.
    """
    
    def __init__(self, name: str, initial_epoch: datetime, tle_line1: str, tle_line2: str, norad_id: Optional[str] = None) -> None:
        """Initializes a Spacecraft instance.
        
        Args:
            name (str): the display name of the object. Defaults to 'Sun'.
            tle_line1 (str): first line of the TLE for propagation.
            tle_line1 (str): second line of the TLE for propagation.
            norad_id (Optional[str]): the NORAD ID of the spacecraft. Defaults to None.
            initial_epoch (datetime): the first epoch from which to derive values.
            
        Returns:
            None

        Raises:
            TypeError: if arguments are not of expected type.
        
        """
        super().__init__(name, initial_epoch, tle_line1, tle_line2, norad_id)
        self.body_to_gcrs_rotation = None
        self.sensors = []

    # Position Vectors

    @property
    def spacecraft_earth_vector_teme(self) -> np.ndarray:
        """
        Returns the vector pointing from the spacecraft to Earth in the True Equator Mean Equinox (TEME) reference frame.
        
        Returns:
            np.ndarray: A numpy array representing the spacecraft-to-Earth vector in TEME coordinates.
        """
        return -1 * self.position_vector_teme

    @property
    def spacecraft_earth_vector_gcrs(self) -> np.ndarray:
        """
        Returns the vector pointing from the spacecraft to Earth in the Geocentric Celestial Reference System (GCRS) reference frame.
        
        Returns:
            np.ndarray: A numpy array representing the spacecraft-to-Earth vector in GCRS coordinates.
        """
        return -1 * self.position_vector_gcrs

    @property
    def spacecraft_earth_vector_lvlh(self) -> np.ndarray:
        """
        Returns the vector pointing from the spacecraft to Earth in the Local Vertical Local Horizontal (LVLH) reference frame.
        
        Returns:
            np.ndarray: A numpy array representing the spacecraft-to-Earth vector in LVLH coordinates.
        """

        position_gcrs, velocity_gcrs = self.position_and_velocity_vectors_gcrs
        gcrs_lvlh_rotation = gcrs_to_lvlh_rotation(position_gcrs, velocity_gcrs)
        return gcrs_lvlh_rotation.apply(self.spacecraft_earth_vector_gcrs)

    @property
    def spacecraft_sun_vector_teme(self) -> np.ndarray:
        """
        Returns the vector pointing from the spacecraft to the Sun in the True Equator Mean Equinox (TEME) reference frame.
        
        Returns:
            np.ndarray: A numpy array representing the spacecraft-to-Sun vector in TEME coordinates.
        """

        earth_sun_vector_teme = get_earth_sun_vector_teme_at_epoch(self.epoch)
        return earth_sun_vector_teme - self.position_vector_teme

    @property
    def spacecraft_sun_vector_gcrs(self) -> np.ndarray:
        """
        Returns the vector pointing from the spacecraft to the Sun in the Geocentric Celestial Reference System (GCRS) reference frame.
        
        Returns:
            np.ndarray: A numpy array representing the spacecraft-to-Sun vector in GCRS coordinates.
        """

        earth_sun_vector_gcrs = get_earth_sun_vector_gcrs_at_epoch(self.epoch)
        return earth_sun_vector_gcrs - self.position_vector_gcrs

    @property
    def spacecraft_sun_vector_lvlh(self) -> np.ndarray:
        """
        Returns the vector pointing from the spacecraft to the Sun in the Local Vertical Local Horizontal (LVLH) reference frame.
        
        Returns:
            np.ndarray: A numpy array representing the spacecraft-to-Sun vector in LVLH coordinates.
        """

        position_gcrs, velocity_gcrs = self.position_and_velocity_vectors_gcrs
        gcrs_lvlh_rotation = gcrs_to_lvlh_rotation(position_gcrs, velocity_gcrs)
        return gcrs_lvlh_rotation.apply(self.spacecraft_sun_vector_gcrs)

    # Sensor things

    def add_sensor(self, sensor: BasicSensor) -> None:
        """
        Adds a sensor to the spacecraft.

        Args:
            sensor (BasicSensor): The sensor to be added to the spacecraft.

        Raises:
            TypeError: If `sensor` is not an instance of `BasicSensor`.
        """

        if not isinstance(sensor, BasicSensor):
            raise TypeError(f"arg 'sensor' must be of type BasicSensor, not {type(sensor)}")
        self.sensors.append(sensor)

    def check_sensor_sun_exclusion_zones(self, reference_frame: Optional[str] = "GCRS", body_to_reference_frame: Optional[R] = None) -> list[str]:
        """
        Checks which sensors on the spacecraft have their sun exclusion zones violated in the specified reference frame.

        Args:
            reference_frame (Optional[str]): The reference frame in which to check exclusion zones. Defaults to "GCRS".
            body_to_reference_frame (Optional[R]): A rotation object specifying the spacecraft body to the specified reference frame transformation.

        Returns:
            list[str]: A list of sensor names that have their sun exclusion zones violated.

        Raises:
            TypeError: If `reference_frame` is not a string.
            ValueError: If `reference_frame` is not one of the accepted values ("GCRS", "TEME", "LVLH").
        """
        
        # Argument checking
        if not isinstance(reference_frame, str):
            raise TypeError(f"arg 'reference_frame' must be of type str, not {type(reference_frame)}")
        if reference_frame not in ["GCRS", "TEME", "LVLH"]:
            raise ValueError(f"'reference_frame' must be one of ['GCRS', 'TEME', 'LVLH'] (gave {reference_frame}")

        sun_exclusion_zones = []
        
        match reference_frame:
            case "GCRS":
                for sensor in self.sensors:
                    if sensor.sun_exclusion_zone_violated(self.get_body_to_gcrs_rotation(), self.spacecraft_sun_vector_gcrs):
                        sun_exclusion_zones.append(sensor.name)
            case "TEME":
                for sensor in self.sensors:
                    if sensor.sun_exclusion_zone_violated(body_to_reference_frame, self.spacecraft_sun_vector_teme):
                        sun_exclusion_zones.append(sensor.name)
            case "LVLH":
                for sensor in self.sensors:
                    if sensor.sun_exclusion_zone_violated(self.get_body_to_lvlh_rotation(), self.spacecraft_sun_vector_lvlh):
                        sun_exclusion_zones.append(sensor.name)
        
        return sun_exclusion_zones

    def check_sensor_earth_exclusion_zones(self, reference_frame: Optional[str] = "TEME", body_to_reference_frame: Optional[R] = None) -> list[str]:
        """
        Checks which sensors on the spacecraft have their earth exclusion zones violated in the specified reference frame.

        Args:
            reference_frame (Optional[str]): The reference frame in which to check exclusion zones. Defaults to "TEME".
            body_to_reference_frame (Optional[R]): A rotation object specifying the spacecraft body to the specified reference frame transformation.

        Returns:
            list[str]: A list of sensor names that have their earth exclusion zones violated.

        Raises:
            TypeError: If `reference_frame` is not a string.
            ValueError: If `reference_frame` is not one of the accepted values ("GCRS", "TEME", "LVLH").
        """
        
        # Argument checking
        if not isinstance(reference_frame, str):
            raise TypeError(f"arg 'reference_frame' must be of type str, not {type(reference_frame)}")
        if reference_frame not in ["GCRS", "TEME", "LVLH"]:
            raise ValueError(f"'reference_frame' must be one of ['GCRS', 'TEME', 'LVLH'] (gave {reference_frame}")
        
        earth_exclusion_zones = []

        match reference_frame:
            case "GCRS":
                for sensor in self.sensors:
                    if sensor.earth_exclusion_zone_violated(self.get_body_to_gcrs_rotation(), self.spacecraft_earth_vector_gcrs):
                        earth_exclusion_zones.append(sensor.name)
            case "TEME":
                for sensor in self.sensors:
                    if sensor.earth_exclusion_zone_violated(body_to_reference_frame, self.spacecraft_earth_vector_teme):
                        earth_exclusion_zones.append(sensor.name)
            case "LVLH":
                for sensor in self.sensors:
                    if sensor.earth_exclusion_zone_violated(self.get_body_to_lvlh_rotation(), self.spacecraft_earth_vector_lvlh):
                        earth_exclusion_zones.append(sensor.name)

        return earth_exclusion_zones

    def check_sensor_sun_and_earth_exclusion_zones(self, reference_frame: Optional[str] = "GCRS", body_to_reference_frame: Optional[R] = None) -> tuple[list[str], list[str]]:
        """
        Checks which sensors on the spacecraft have either their sun or earth exclusion zones violated in the specified reference frame.

        Args:
            reference_frame (Optional[str]): The reference frame in which to check exclusion zones. Defaults to "GCRS".
            body_to_reference_frame (Optional[R]): A rotation object specifying the spacecraft body to the specified reference frame transformation.

        Returns:
            tuple[list[str], list[str]]: Two lists containing the names of sensors that have their sun and earth exclusion zones violated, respectively.

        Raises:
            TypeError: If `reference_frame` is not a string.
            ValueError: If `reference_frame` is not one of the accepted values ("GCRS", "TEME", "LVLH").
        """
        
        # Argument checking
        if not isinstance(reference_frame, str):
            raise TypeError(f"arg 'reference_frame' must be of type str, not {type(reference_frame)}")
        if reference_frame not in ["GCRS", "TEME", "LVLH"]:
            raise ValueError(f"'reference_frame' must be one of ['GCRS', 'TEME', 'LVLH'] (gave {reference_frame}")

        sun_exclusion_zones = []
        earth_exclusion_zones = []

        if reference_frame == "TEME":
            sun_exclusion_zones = self.check_sensor_sun_exclusion_zones(reference_frame, body_to_reference_frame)
            earth_exclusion_zones = self.check_sensor_earth_exclusion_zones(reference_frame, body_to_reference_frame)
        
        sun_exclusion_zones = self.check_sensor_sun_exclusion_zones(reference_frame)
        earth_exclusion_zones = self.check_sensor_earth_exclusion_zones(reference_frame)

        return sun_exclusion_zones, earth_exclusion_zones

    # Rotation things

    def set_body_to_gcrs_rotation(self, body_to_gcrs_rotation: R) -> None:
        """
        Sets the rotation from the spacecraft body frame to the Geocentric Celestial Reference System (GCRS) frame.

        Args:
            body_to_gcrs_rotation (R): A scipy Rotation object representing the rotation from body to GCRS frame.

        Raises:
            TypeError: If `body_to_gcrs_rotation` is not an instance of scipy's Rotation class.
        """

        if not isinstance(body_to_gcrs_rotation, R):
            raise TypeError(f"arg 'body_to_gcrs_rotation' must be of type R, not {type(body_to_gcrs_rotation)}")
        self.body_to_gcrs_rotation = body_to_gcrs_rotation

    def get_body_to_gcrs_rotation(self) -> R:
        """
        Retrieves the spacecraft's rotation from body frame to GCRS frame.

        Returns:
            R: A scipy Rotation object representing the rotation from body to GCRS frame.

        Raises:
            AttributeError: If the body to GCRS rotation has not been set.
        """

        if self.body_to_gcrs_rotation is None:
            raise AttributeError(f"attribute 'self.body_to_gcrs_rotation' of Spacecraft '{self.name}' has not yet been set.")
        return self.body_to_gcrs_rotation

    def get_body_to_lvlh_rotation(self) -> R:
        """
        Computes and retrieves the spacecraft's rotation from body frame to Local Vertical Local Horizontal (LVLH) frame.

        Returns:
            R: A scipy Rotation object representing the rotation from body to LVLH frame.

        Raises:
            AttributeError: If the body to GCRS rotation has not been set.
        """

        if self.body_to_gcrs_rotation is None:
            raise AttributeError(f"attribute 'self.body_to_gcrs_rotation' of Spacecraft '{self.name}' has not yet been set.")
        gcrs_to_lvlh_rotation_obj = gcrs_to_lvlh_rotation(self.position_vector_gcrs, self.velocity_vector_gcrs)
        return self.body_to_gcrs_rotation * gcrs_to_lvlh_rotation_obj
