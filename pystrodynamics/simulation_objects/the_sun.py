# Standard library imports
from datetime import datetime, timezone

# Third party imports
import numpy as np

# Local imports
from pystrodynamics.simulation_objects.simulation_object import SimulationObject
from pystrodynamics.utils.sun import get_earth_sun_vector_gcrs_at_epoch, get_earth_sun_vector_teme_at_epoch

class TheSun(SimulationObject):
    """Simulation object for the Sun."""

    def __init__(self, name: str, initial_epoch: datetime) -> None:
        """Initializes a TheSun instance.
        
        Args:
            name (str): the display name of the object. Defaults to 'Sun'.
            initial_epoch (datetime): the first epoch from which to derive values.
            
        Returns:
            None

        Raises:
            TypeError: if arguments are not of expected type.
        
        """
        if not isinstance(initial_epoch, datetime):
            raise TypeError(f"arg initial_epoch must be of type datetime, not {str(type(initial_epoch))}")
        super().__init__(name)
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
            raise TypeError(f"arg epoch must be of type datetime, not {str(type(epoch))}")
        epoch = epoch.replace(tzinfo=timezone.utc)
        self.epoch = epoch

    @property
    def earth_sun_vector_gcrs(self) -> np.ndarray:
        """The vector from the Earth to the Sun in GCRS frame.

        Args:
            None

        Returns:
            earth_sun_vector_gcrs (np.ndarray): Earth-Sun vector in GCRS.

        """
        return get_earth_sun_vector_gcrs_at_epoch(self.epoch)

    @property
    def earth_sun_vector_teme(self) -> np.ndarray:
        """The vector from the Earth to the Sun in TEME frame.

        Note: much slower than earth_sun_vector_gcrs due to the necessary frame transformation.

        Args:
            None

        Returns:
            earth_sun_vector_teme (np.ndarray): Earth-Sun vector in TEME.

        """
        return get_earth_sun_vector_teme_at_epoch(self.epoch)

