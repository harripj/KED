import numpy as np
from scipy import constants

from .reciprocal_lattice import theta_to_k
from .utils import DTYPE


def electron_wavelength(kV, dtype=DTYPE):
    """
    Calculate the electron wavelength with accelerating voltage V (kV).

    Parameters
    ----------
    V: float
        Acceleraing voltage in kV.

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


def calculate_pixel_size(camera_diameter, camera_length, shape, wavelength):
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
