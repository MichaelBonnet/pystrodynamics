# Standard library imports

# Third party imports
import numpy as np
import pytest

# Local imports
from pystrodynamics.utils.rotations import gcrs_to_lvlh_rotation

def test_gcrs_to_lvlh_rotation_with_validation_data():
    pass

def test_gcrs_to_lvlh_rotation_bad_vectors():
    # Bad position vector size
    with pytest.raises(TypeError):
        gcrs_to_lvlh_rotation(np.random.rand(2), np.random.rand(3))

    # Bad position vector type
    with pytest.raises(TypeError):
        gcrs_to_lvlh_rotation(None, np.random.rand(3))

    # Bad velocity vector size
    with pytest.raises(TypeError):
        gcrs_to_lvlh_rotation(np.random.rand(3), np.random.rand(2))

    # Bad velocity vector type
    with pytest.raises(TypeError):
        gcrs_to_lvlh_rotation(np.random.rand(3), None)
