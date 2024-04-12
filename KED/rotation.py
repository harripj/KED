import numba
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from orix.quaternion import Quaternion, Rotation
from orix.vector import AxAngle, Vector3d

# constant for the rotation order between ASTAR and scipy
EULER_ROTATION_ORDER = "ZXZ"


def calculate_rotation_vector(v1: ArrayLike, v2: ArrayLike) -> Rotation:
    """
    Calculate the rotation vector that rotates v1 onto v2.
    This rotation vector is normal to the plane that incorporates the two vectors.

    Parameters
    ----------
    v1, v2: ArrayLike
        The two vectors.

    Returns
    -------
    Rotation:
        The Rotation object that rotates v1 onto v2.

    """
    norm = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    return Rotation.from_neo_euler(AxAngle(norm * angle / np.linalg.norm(norm)))


@numba.njit
def _rotate_coords_2d(coords: NDArray, angle: float) -> NDArray:
    """
    Rotate a set of coordinates by an angle.

    Parameters
    ----------
    coords: (N, 2) ndarray
        Coordinates to rotate (about (0, 0)).
    angles: float
        Angle to rotate in radians.

    Returns
    -------
    rotated: (N, 2) ndarray
        The rotated coordinates.
    """
    # define rotation matrix
    R = np.array(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
        dtype=coords.dtype,
    )
    return np.dot(coords, R)


@numba.njit
def generate_rotated_template(
    coords: NDArray,
    angle: float,
    scale_factor: float,
    shift: NDArray,
) -> NDArray:
    """
    Rotate a template rotated by angle.

    Parameters
    ----------
    coords: (N, 2) ndarray
        Coordinates to rotate (about (0, 0)).
    angle: float
        Angle to rotate in radians.
    scale_factor: float
        The coordinate scaling factor in 1/Angstrom.
    shift: (2,) ndarray
        Shift to apply after rotation around (0, 0).

    Returns
    -------
    rotated: (N, 2) ndarray
        The rotated coordinates.
    """
    return _rotate_coords_2d(coords / scale_factor, angle) + shift
