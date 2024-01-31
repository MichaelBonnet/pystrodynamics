# Standard library imports

# Third part imports
import numpy as np
from scipy.spatial.transform import Rotation as R

# Local imports
from pystrodynamics.simulation_objects.spacecraft_modules.spacecraft_module import SpacecraftModule
from pystrodynamics.utils.math import angle_between_vectors, norm

class BasicSensor(SpacecraftModule):
    """
    Represents a basic sensor module for a spacecraft, including functionality for detecting targets,
    checking exclusion zones, and determining if a target is within the sensor's field of view and range.

    Attributes:
        boresight_unit_vector (np.ndarray): The unit vector along the sensor's boresight.
        sun_exclusion_angle_deg (float): The minimum angle from the boresight within which the Sun must not be detected.
        earth_exclusion_angle_deg (float): The minimum angle from the boresight within which the Earth must not be detected.
        effective_range_km (float): The maximum distance (in kilometers) from the sensor within which a target can be detected.
        field_of_view_half_angle_deg (float): Half the angle of the sensor's field of view in degrees.
    """

    def __init__(self,
                 name: str,
                 boresight_unit_vector: np.ndarray,
                 sun_exclusion_angle_deg: float,
                 earth_exclusion_angle_deg: float,
                 effective_range_km: float,
                 field_of_view_half_angle_deg: float) -> None:
        """
        Initializes a BasicSensor object with specified parameters.

        Args:
            name (str): The name of the sensor.
            boresight_unit_vector (np.ndarray): The unit vector along the sensor's boresight.
            sun_exclusion_angle_deg (float): The sun exclusion angle in degrees.
            earth_exclusion_angle_deg (float): The earth exclusion angle in degrees.
            effective_range_km (float): The effective range of the sensor in kilometers.
            field_of_view_half_angle_deg (float): Half of the field of view angle in degrees.

        Raises:
            TypeError: If any argument is not of the expected type.
        """

        # Argument checking
        if not isinstance(boresight_unit_vector, np.ndarray):
            raise TypeError(f"arg 'boresight_unit_vector' must be of type np.ndarray, not {type(boresight_unit_vector)}")
        if not isinstance(sun_exclusion_angle_deg, float):
            raise TypeError(f"arg 'sun_exclusion_angle_deg' must be of type float, not {type(sun_exclusion_angle_deg)}")
        if not isinstance(earth_exclusion_angle_deg, float):
            raise TypeError(f"arg 'earth_exclusion_angle_deg' must be of type float, not {type(earth_exclusion_angle_deg)}")
        if not isinstance(effective_range_km, float):
            raise TypeError(f"arg 'effective_range_km' must be of type float, not {type(effective_range_km)}")
        if not isinstance(field_of_view_half_angle_deg, float):
            raise TypeError(f"arg 'field_of_view_half_angle_deg' must be of type float, not {type(field_of_view_half_angle_deg)}")
        
        super().__init__(name)

        self.boresight_unit_vector = boresight_unit_vector
        self.sun_exclusion_angle_deg = sun_exclusion_angle_deg
        self.earth_exclusion_angle_deg = earth_exclusion_angle_deg
        self.effective_range_km = effective_range_km
        self.field_of_view_half_angle_deg = field_of_view_half_angle_deg

    def target_angle_from_boresight(self, body_to_frame_rotation: R, target_vector: np.ndarray) -> float:
        """
        Calculates the angle between the sensor's boresight and a target vector.

        Args:
            body_to_frame_rotation (R): The rotation from body to reference frame.
            target_vector (np.ndarray): The vector to the target in the reference frame.

        Returns:
            float: The angle between the boresight and the target vector in degrees.

        Raises:
            TypeError: If any argument is not of the expected type.
        """
        # Argument checking
        if not isinstance(body_to_frame_rotation, R):
            raise TypeError(f"arg 'body_to_frame_rotation' must be of type R, not {type(body_to_frame_rotation)}")
        if not isinstance(target_vector, R):
            raise TypeError(f"arg 'target_vector' must be of type np.ndarray, not {type(target_vector)}")
        
        boresight_unit_vector_in_frame = body_to_frame_rotation.apply(self.boresight_unit_vector)
        return angle_between_vectors(boresight_unit_vector_in_frame, target_vector, "degrees")

    def exclusion_zone_violated(self, body_to_frame_rotation: R, exclusion_vector: np.ndarray, exclusion_angle: float) -> bool:
        """
        Determines if the target violates a specified exclusion zone.

        Args:
            body_to_frame_rotation (R): The rotation from body to reference frame.
            exclusion_vector (np.ndarray): The vector to the object to check against the exclusion zone.
            exclusion_angle (float): The angle defining the exclusion zone.

        Returns:
            bool: True if the exclusion zone is violated, False otherwise.

        Raises:
            TypeError: If any argument is not of the expected type.
        """
        # Argument checking
        if not isinstance(body_to_frame_rotation, R):
            raise TypeError(f"arg 'body_to_frame_rotation' must be of type R, not {type(body_to_frame_rotation)}")
        if not isinstance(exclusion_vector, R):
            raise TypeError(f"arg 'exclusion_vector' must be of type np.ndarray, not {type(exclusion_vector)}")
        if not isinstance(exclusion_angle, R):
            raise TypeError(f"arg 'exclusion_angle' must be of type float, not {type(exclusion_angle)}")
        
        return self.target_angle_from_boresight(body_to_frame_rotation, exclusion_vector) < exclusion_angle

    def earth_exclusion_zone_violated(self, body_to_frame_rotation: R, spacecraft_to_earth_vector: np.ndarray) -> bool:
        """
        Determines if the Earth violates the sensor's Earth exclusion zone.

        Args:
            body_to_frame_rotation (R): The rotation from body to reference frame.
            spacecraft_to_earth_vector (np.ndarray): The vector from the spacecraft to Earth in the reference frame.

        Returns:
            bool: True if the Earth is within the sensor's Earth exclusion zone, False otherwise.

        Raises:
            TypeError: If any argument is not of the expected type.
        """
        # Argument checking
        if not isinstance(body_to_frame_rotation, R):
            raise TypeError(f"arg 'body_to_frame_rotation' must be of type R, not {type(body_to_frame_rotation)}")
        if not isinstance(spacecraft_to_earth_vector, R):
            raise TypeError(f"arg 'spacecraft_to_earth_vector' must be of type np.ndarray, not {type(spacecraft_to_earth_vector)}")
        
        return self.exclusion_zone_violated(body_to_frame_rotation, spacecraft_to_earth_vector, self.earth_exclusion_angle_deg)
    
    def sun_exclusion_zone_violated(self, body_to_frame_rotation: R, spacecraft_to_sun_vector: np.ndarray) -> bool:
        """
        Determines if the Sun violates the sensor's Sun exclusion zone.

        Args:
            body_to_frame_rotation (R): The rotation from body to reference frame.
            spacecraft_to_sun_vector (np.ndarray): The vector from the spacecraft to the Sun in the reference frame.

        Returns:
            bool: True if the Sun is within the sensor's Sun exclusion zone, False otherwise.

        Raises:
            TypeError: If any argument is not of the expected type.
        """
        # Argument checking
        if not isinstance(body_to_frame_rotation, R):
            raise TypeError(f"arg 'body_to_frame_rotation' must be of type R, not {type(body_to_frame_rotation)}")
        if not isinstance(spacecraft_to_sun_vector, R):
            raise TypeError(f"arg 'spacecraft_to_sun_vector' must be of type np.ndarray, not {type(spacecraft_to_sun_vector)}")
        
        return self.exclusion_zone_violated(body_to_frame_rotation, spacecraft_to_sun_vector, self.sun_exclusion_angle_deg)

    def target_in_field_of_view(self, body_to_frame_rotation: R, spacecraft_to_target_vector: np.ndarray) -> bool:
        """
        Determines if a target is within the sensor's field of view.

        Args:
            body_to_frame_rotation (R): The rotation from body to reference frame.
            spacecraft_to_target_vector (np.ndarray): The vector from the spacecraft to the target in the reference frame.

        Returns:
            bool: True if the target is within the sensor's field of view, False otherwise.

        Raises:
            TypeError: If any argument is not of the expected type.
        """
        # Argument checking
        if not isinstance(body_to_frame_rotation, R):
            raise TypeError(f"arg 'body_to_frame_rotation' must be of type R, not {type(body_to_frame_rotation)}")
        if not isinstance(spacecraft_to_target_vector, R):
            raise TypeError(f"arg 'spacecraft_to_target_vector' must be of type np.ndarray, not {type(spacecraft_to_target_vector)}")
        
        return self.target_angle_from_boresight(body_to_frame_rotation, spacecraft_to_target_vector) < self.field_of_view_half_angle_deg

    def target_in_range(self, spacecraft_to_target_vector: np.ndarray) -> bool:
        """
        Determines if a target is within the sensor's effective range.

        Args:
            spacecraft_to_target_vector (np.ndarray): The vector from the spacecraft to the target.

        Returns:
            bool: True if the target is within the sensor's effective range, False otherwise.

        Raises:
            TypeError: If any argument is not of the expected type.
        """
        # Argument checking
        if not isinstance(spacecraft_to_target_vector, R):
            raise TypeError(f"arg 'spacecraft_to_target_vector' must be of type np.ndarray, not {type(spacecraft_to_target_vector)}")
        
        return norm(spacecraft_to_target_vector) <= self.effective_range_km

    def target_is_accessible(self, body_to_frame_rotation: R, spacecraft_to_target_vector: np.ndarray) -> bool:
        """
        Determines if a target is both within the field of view and effective range of the sensor.

        Args:
            body_to_frame_rotation (R): The rotation from body to reference frame.
            spacecraft_to_target_vector (np.ndarray): The vector from the spacecraft to the target.

        Returns:
            bool: True if the target is accessible (within both the field of view and effective range), False otherwise.

        Raises:
            TypeError: If any argument is not of the expected type.
        """
        # Argument checking
        if not isinstance(body_to_frame_rotation, R):
            raise TypeError(f"arg 'body_to_frame_rotation' must be of type R, not {type(body_to_frame_rotation)}")
        if not isinstance(spacecraft_to_target_vector, R):
            raise TypeError(f"arg 'spacecraft_to_target_vector' must be of type np.ndarray, not {type(spacecraft_to_target_vector)}")

        return (self.target_in_field_of_view(body_to_frame_rotation, spacecraft_to_target_vector) and self.target_in_range(spacecraft_to_target_vector))
