import numpy as np

def norm(x: np.ndarray) -> float:
    return np.linalg.norm(x, axis=0)

def unit_vector(x: np.ndarray) -> np.ndarray:
    return x / norm(x)

def angle_between_vectors(a: np.ndarray, b: np.ndarray, units: str = "radians") -> float:
    n_a = np.linalg.norm(a, axis=0)
    n_b = np.linalg.norm(b, axis=0)
    y = np.linalg.norm(n_b * a - n_a * b, axis=0)
    x = np.linalg.norm(n_b * a + n_a * b, axis=0)
    if units == "degrees":
        return np.rad2deg(2 * np.arctan2(y, x))
    elif units == "radians":
        return 2 * np.arctan2(y, x)
