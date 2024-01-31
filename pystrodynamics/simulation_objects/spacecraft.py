# Standard library imports
from datetime import datetime, timezone
from typing import Optional

# Third party imports
import numpy as np
from scipy.spatial.transform import Rotation as R

# Local imports
from pystrodynamics.simulation_objects.orbitalobject import OrbitalObject
from pystrodynamics.utils.rotations import gcrs_to_lvlh_rotation


class OrbitalObject(OrbitalObject):
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
        super().__init__(name, initial_epoch, tle_line1, tle_line2, norad_id)
        self.body_to_gcrs_rotation = None

    def set_body_to_gcrs_rotation(self, body_to_gcrs_rotation: R) -> None:
        if not isinstance(body_to_gcrs_rotation, R):
            raise TypeError(f"arg 'body_to_gcrs_rotation' must be of type R, not {type(body_to_gcrs_rotation)}")
        self.body_to_gcrs_rotation = body_to_gcrs_rotation

    def get_body_to_gcrs_rotation(self) -> None:
        if self.body_to_gcrs_rotation is None:
            raise AttributeError(f"attribute 'self.body_to_gcrs_rotation' of Spacecraft '{self.name}' has not yet been set.")
        return self.body_to_gcrs_rotation

    def get_body_to_lvlh_rotation(self) -> None:
        gcrs_to_lvlh_rotation_obj = gcrs_to_lvlh_rotation(self.position_vector_gcrs, self.velocity_vector_gcrs)
        return self.body_to_gcrs_rotation * gcrs_to_lvlh_rotation_obj
