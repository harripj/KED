import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from scipy import constants

from .reciprocal_lattice import theta_to_k
from .utils import DTYPE


def electron_wavelength(kV: float, dtype: DTypeLike = DTYPE) -> float:
    """
    Calculate the electron wavelength with accelerating voltage V (kV).

    Parameters
    ----------
    kV: float
        Accelerating voltage in kV.

    Returns
    -------
    lambda: float
        Electron wavelength in Angstroms.
    """

    V = kV * constants.kilo  # conversion between kV and V
    eV = constants.elementary_charge * V

    wavelength = (
        constants.Planck
        * constants.c
        / (eV * (2 * constants.electron_mass * constants.c**2 + eV)) ** 0.5
    )
    return dtype(wavelength / constants.angstrom)


def calculate_pixel_size(
    camera_diameter: float,
    camera_length: float,
    shape: ArrayLike,
    wavelength: float,
) -> float:
    """Calculate the pixel size in reciprocal space.

    Parameters
    ----------
    camera_diameter: float
        Physical size of the camera in m.
    camera_length: float
        The indicated camera length in m.
    shape: int or array-like
        The size of the data (including any binning).
    wavelength: float
        The electron wavelength in Angstroms.

    Returns
    -------
    pixel_size: float
        The pixel size in 1/Angstroms.
    """

    radius_camera = camera_diameter / 2.0
    camera_max_angle = np.tan(radius_camera / camera_length)

    if isinstance(shape, (int, np.integer)):
        shape = (shape,)

    # shape / 2 as radius is measured from center
    pixel_size_radians = camera_max_angle / (min(shape) / 2.0)
    pixel_size_k = theta_to_k(pixel_size_radians, wavelength)
    return pixel_size_k
