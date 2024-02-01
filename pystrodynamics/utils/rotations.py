"""Functions that provide rotations between reference frames."""

# Standard library imports

# Third-party imports
import numpy as np
from scipy.spatial.transform import Rotation as R

# Local imports
from pystrodynamics.utils.math import unit_vector

def gcrs_to_lvlh_rotation(position_vector_gcrs: np.ndarray, velocity_vector_gcrs: np.ndarray) -> R:
    """Compute the rotation from the GCRS frame 
    to the Local Vertical Local Horizontal (LVLH) frame.

    The LVLH frame is a local frame centered at the satellite with its
    z-axis pointing towards the center of the Earth, x-axis in the direction
    of the satellite's velocity, and y-axis completing the right-handed system.

    Args:
        position_vector_gcrs (np.ndarray): A 3-element numpy array representing the position 
                              vector of the satellite in GCRS frame.
        velocity_vector_gcrs (np.ndarray): A 3-element numpy array representing the velocity 
                              vector of the satellite in GCRS frame.

    Returns:
        gcrs_to_lvlh_rotation (R): A scipy.spatial.transform.Rotation object representing the 
                                   rotation matrix from GCRS frame to LVLH frame.

    Raises:
        TypeError: if an argument is not of expected type.
        IndexError: if either vector is not of size 3.

    """
    # Argument checking
    if not isinstance(position_vector_gcrs, np.ndarray):
        raise TypeError(f"arg 'position_vector_gcrs' must be of type np.ndarray, not {type(position_vector_gcrs)}")
    if not isinstance(velocity_vector_gcrs, np.ndarray):
        raise TypeError(f"arg 'velocity_vector_gcrs' must be of type np.ndarray, not {type(velocity_vector_gcrs)}")
    if len(position_vector_gcrs) != 3:
        raise IndexError(f"'position_vector_gcrs' must be of length 3, not {len(position_vector_gcrs)}")
    if len(velocity_vector_gcrs) != 3:
        raise IndexError(f"'velocity_vector_gcrs' must be of length 3, not {len(velocity_vector_gcrs)}")

    r_v_cross = np.cross(position_vector_gcrs, velocity_vector_gcrs)
    x = unit_vector(position_vector_gcrs)
    z = r_v_cross / unit_vector(r_v_cross)
    y = np.cross(x, z)
    rotation_matrix = np.stack(x, y, z)
    gcrs_to_lvlh_rotation = R.from_matrix(rotation_matrix)
    return gcrs_to_lvlh_rotation