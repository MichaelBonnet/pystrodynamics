# Standard library imports
from datetime import datetime, timezone
from typing import Optional

# Third party imports
import numpy as np
from scipy.spatial.transform import Rotation as R

# Local imports
from pystrodynamics.simulation_objects.simulation_object import SimulationObject
from pystrodynamics.utils.sun import get_earth_sun_vector_teme_at_epoch
from pystrodynamics.utils.eclipse import is_in_eclipse
from pystrodynamics.utils.propagation import tle_and_epoch_to_state_vectors, state_vectors_to_coe


class OrbitalObject(SimulationObject):
    """Simulation object for things orbiting Earth."""
    
    def __init__(self, name: str, initial_epoch: datetime, tle_line1: str, tle_line2: str, norad_id: Optional[str] = None) -> None:
        """Initializes an OrbitalObject instance.
        
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
        # Argument checking
        if not isinstance(initial_epoch, datetime):
            raise TypeError(f"arg 'initial_epoch' must be of type datetime, not {type(initial_epoch)}")
        
        super().__init__(name)
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        self.norad_id = norad_id
        self.update_state(initial_epoch)

    def update_state(self, epoch: datetime) -> None:
        """Updates the state of the Sun object with information for the new epoch.

        Args:
            epoch (datetime): new epoch from which to derive values.

        Returns:
            None

        Raises:
            TypeError: if arguments are not of expected type.

        """
        if not isinstance(epoch, datetime):
            raise TypeError(f"arg 'epoch' must be of type datetime, not {type(epoch)}")
        epoch = epoch.replace(tzinfo=timezone.utc)
        self.epoch = epoch
        self.__position_vector_teme, self.__velocity_vector_teme = tle_and_epoch_to_state_vectors(self.tle_line1, self.tle_line2, self.epoch, "TEME")

    @property
    def classical_orbital_elements(self, reference_frame: Optional[str] = "TEME"):
        # Argument checking
        if not isinstance(reference_frame, str):
            raise TypeError(f"arg 'reference_frame' must be of type str, not {type(reference_frame)}")
        if reference_frame not in ["GCRS", "TEME"]:
            raise ValueError(f"'reference_frame' must be one of ['GCRS', 'TEME'] (gave {reference_frame}")

        match reference_frame:
            case "TEME":
                return state_vectors_to_coe(self.position_vector_teme, self.velocity_vector_teme)
            case "GCRS":
                position_gcrs, velocity_gcrs = self.position_and_velocity_vectors_gcrs
                return state_vectors_to_coe(position_gcrs, velocity_gcrs)

    @property
    def position_vector_teme(self) -> np.ndarray:
        """The position vector of the object in TEME frame.

        Args:
            None

        Returns:
            position_vector_teme (np.ndarray): the position vector of the object in TEME.

        """
        return self.__position_vector_teme

    @property
    def velocity_vector_teme(self) -> np.ndarray:
        """The velocity vector of the object in TEME frame.

        Args:
            None

        Returns:
            velocity_vector_teme (np.ndarray): the velocity vector of the object in TEME.

        """
        return self.__velocity_vector_teme

    @property
    def position_vector_gcrs(self) -> np.ndarray:
        """The position vector of the object in GCRS frame.

        Args:
            None

        Returns:
            position_vector_gcrs (np.ndarray): the position vector of the object in GCRS.

        """
        position_vector_gcrs, _ = tle_and_epoch_to_state_vectors(self.tle_line1, self.tle_line2, self.epoch, "GCRS")
        return position_vector_gcrs

    @property
    def velocity_vector_gcrs(self) -> np.ndarray:
        """The velocity vector of the object in GCRS frame.

        Args:
            None

        Returns:
            velocity_vector_gcrs (np.ndarray): the velocity vector of the object in GCRS.

        """
        _, velocity_vector_gcrs = tle_and_epoch_to_state_vectors(self.tle_line1, self.tle_line2, self.epoch, "GCRS")
        return velocity_vector_gcrs

    @property
    def position_and_velocity_vectors_gcrs(self) -> np.ndarray:
        """The position and velocity vectors of the object in GCRS frame.

        Args:
            None

        Returns:
            position_vector_gcrs (np.ndarray): the position vector of the object in GCRS.
            velocity_vector_gcrs (np.ndarray): the velocity vector of the object in GCRS.

        """
        return tle_and_epoch_to_state_vectors(self.tle_line1, self.tle_line2, self.epoch, "GCRS")

    @property
    def is_in_eclipse(self) -> bool:
        """Whether or not the object is in eclipse."""
        return is_in_eclipse(self.position_vector_teme, get_earth_sun_vector_teme_at_epoch(self.epoch))
