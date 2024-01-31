"""Functions for checking if Earth-orbiting objects are in the umbra or penumbra of Earth - that is, eclipse."""

import numpy as np

from pystrodynamics.utils.math import norm, angle_between_vectors
from pystrodynamics.utils.constants import sun_radius_km, earth_radius_km

def check_object_shadows(earth_object_position_vector_eci: np.ndarray, earth_sun_position_vector_eci: np.ndarray) -> tuple[bool, bool]:
    """Checks if an object is in umbra and/or penumbra.

    Args:
        earth_object_position_vector_eci (np.ndarray): the Earth-Object vector in an ECI frame.
        earth_sun_position_vector_eci (np.ndarray): the Earth-Sun vector in an ECI frame.

    Returns:
        in_umbra (bool): whether the object is in the umbra.
        in_penumbra (bool): whether the object is in the penumbra.
    
    Raises:
        TypeError: if arguments are not of expected types.

    """
    # Argument checking
    if not isinstance(earth_object_position_vector_eci, np.ndarray):
        raise TypeError(f"arg 'earth_object_position_vector_eci' must be of type np.ndarray, not {type(earth_object_position_vector_eci)}")
    if not isinstance(earth_sun_position_vector_eci, np.ndarray):
        raise TypeError(f"arg 'earth_sun_position_vector_eci' must be of type np.ndarray, not {type(earth_sun_position_vector_eci)}")
    
    earth_sun_distance = norm(earth_sun_position_vector_eci)
    umbra_angle = np.arctan((sun_radius_km-earth_radius_km)/earth_sun_distance)
    penumbra_angle = np.arctan((sun_radius_km+earth_radius_km)/earth_sun_distance)

    in_umbra = False
    in_penumbra = False
    
    if np.dot(earth_object_position_vector_eci, earth_sun_position_vector_eci) < 0.0:
        angle = angle_between_vectors(-earth_sun_position_vector_eci, earth_object_position_vector_eci)
        sathoriz = norm(earth_object_position_vector_eci)*np.cos(angle)
        satvert  = norm(earth_object_position_vector_eci)*np.sin(angle)
        x = earth_radius_km/np.sin(penumbra_angle)
        penvert = np.tan(penumbra_angle)*(x + sathoriz)
        if satvert <= penvert:
            in_penumbra = True
            y = earth_radius_km/np.sin(umbra_angle)
            umbvert = np.tan(umbra_angle)*(y-sathoriz)
            if satvert <= umbvert:
                in_umbra = True

    return in_penumbra, in_umbra

def is_in_eclipse(earth_object_position_vector_eci: np.ndarray, earth_sun_position_vector_eci: np.ndarray) -> bool:
    """Checks if an object is in eclipse - that is, in either umbra or penumbra.

    Args:
        earth_object_position_vector_eci (np.ndarray): the Earth-Object vector in an ECI frame.
        earth_sun_position_vector_eci (np.ndarray): the Earth-Sun vector in an ECI frame.

    Returns:
        in_eclipse (bool): whether the object is in eclipse.
    
    Raises:
        TypeError: if arguments are not of expected types.

    """
    # Argument checking
    if not isinstance(earth_object_position_vector_eci, np.ndarray):
        raise TypeError(f"arg 'earth_object_position_vector_eci' must be of type np.ndarray, not {type(earth_object_position_vector_eci)}")
    if not isinstance(earth_sun_position_vector_eci, np.ndarray):
        raise TypeError(f"arg 'earth_sun_position_vector_eci' must be of type np.ndarray, not {type(earth_sun_position_vector_eci)}")
    
    in_penumbra, in_umbra = check_object_shadows(earth_object_position_vector_eci, earth_sun_position_vector_eci)
    if in_penumbra or in_umbra:
        return True
    return False