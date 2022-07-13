import logging
from pathlib import Path

from ase.data import atomic_numbers, chemical_symbols
import numpy as np
import pandas as pd
from scipy.constants import Planck, angstrom, electron_mass, electron_volt, epsilon_0

from .structure import get_element_name, get_positions
from .utils import DTYPE

# define default maximum excitation error
S_MAX = 0.1


def read_atomic_scattering_factors():
    """
    Load atomic scattering factors database.

    Data from paper [1]

    Notes
    -----
    [1] Electron atomic scattering factors and scattering potentials of
    crystals, L M Peng. DOI: 10.1016/S0968-4328(99)00033-5

    """

    path = Path(__file__).parent.joinpath("data", "atomic_scattering_factors.txt")
    return pd.read_csv(path)


def read_atomic_scattering_factors_xray():
    """

    Load xray atomic scattering factors database.

    Data from paper [1].

    Notes
    -----
    [1] International Tables for Crystallography: DOI: 10.1107/97809553602060000600
    [2] http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php

    """

    path = Path(__file__).parent.joinpath("data", "atomic_scattering_factors_xray.txt")
    return pd.read_csv(path, delimiter="\s+", skiprows=2)


# load once globally for the rest of the file
ATOMIC_SCATTERING_FACTORS = read_atomic_scattering_factors()
ATOMIC_SCATTERING_FACTORS_XRAY = read_atomic_scattering_factors_xray()


def read_debye_waller_factors_less_than_80K():
    """
    Read Debye-Waller factors which have been parameterized.

    Notes
    -----
    [1] Gao, H. X. & Peng, L.-M. (1999). Acta Cryst. A55, 926-932.
        DOI: 10.1107/S0108767399005176

    """
    path = Path(__file__).parent.joinpath("data", "debye-waller_factors_80K-.txt")
    return pd.read_csv(path, delimiter="\s+", skiprows=1)


def read_debye_waller_factors_greater_than_80K():
    """
    Read Debye-Waller factors which have been parameterized.

    Notes
    -----
    [1] Gao, H. X. & Peng, L.-M. (1999). Acta Cryst. A55, 926-932.
        DOI: 10.1107/S0108767399005176

    """
    path = Path(__file__).parent.joinpath("data", "debye-waller_factors_80K+.txt")
    return pd.read_csv(path, delimiter="\s+", skiprows=1)


# load once globally for the rest of the file
DEBYE_WALLER_SCATTERING_FACTORS_LESS_THAN_80K = (
    read_debye_waller_factors_less_than_80K()
)
DEBYE_WALLER_SCATTERING_FACTORS_GREATER_THAN_80K = (
    read_debye_waller_factors_greater_than_80K()
)


def calculate_debye_waller_factor(element, T=293.0, structure=None):
    """
    Calculate the Debye-Waller factor B for an element at temperature T.

    B is calculated from the sum of polynomials:
        B = a0 + a1T + a2T**2 + a3T**3 + a4T**4

    Parameters
    ----------
    element: str, ase.Atom, diffpy.structure.Atom, int
        Element or identifier.
    T: float
        Temperature in Kelvin.
    structure: None or str
        Structure filter for database.

    Returns
    -------
    B: float
        The Debye-Waller factor B.

    """
    if T <= 80:
        factors = DEBYE_WALLER_SCATTERING_FACTORS_LESS_THAN_80K
    else:
        factors = DEBYE_WALLER_SCATTERING_FACTORS_GREATER_THAN_80K

    # get element name as str
    element = get_element_name(element)

    # isolate (maybe more or less than) the one row
    row = factors[factors["Element"] == element]

    # apply structure filtering
    if structure is not None:
        row = row[row["Structure"].str.lower() == structure.lower()]

    if not len(row):
        temp = {structure if structure is not None else "Any"}
        raise ValueError(f"{element} with structure {temp} not found in database.")
    if not len(row) == 1:
        logging.error(
            f"Multiple element with name {element} found in database: "
            + f"using first input.\n{row}"
        )

    # B is calculated as the sum of polynomials up to order 4
    B = sum(row.iloc[0][f"a{i}"] * T**i for i in range(5))

    return B


def mott_bethe_approximation(fx, Z, q):
    """
    Apply the Mott-Bethe approximation to xray atomic scattering factors
    to compute electron scattering factors.

    Parameters
    ----------
    fx: float, ndarray
        The xray atomic scattering factors.
    Z: int
        Atomic mass of the element.
    q: float, ndarray
        Scattering vector in units 1/Angstrom.

    Returns
    -------
    fe: float, ndarray
        The electron atomic scattering factor.

    """

    A = (
        electron_mass
        * electron_volt**2
        / (32 * np.pi**3 * (Planck / (2 * np.pi)) ** 2 * epsilon_0)
    ) * angstrom

    return A * (Z - fx) / q**2


def calculate_scattering_factor(element, q, use_xray=False, dtype=DTYPE):
    """
    Calculate the atomic scattering factor for atom with scattering
    vector q.

    Parameters
    ----------
    element: str, int, ase.Atom, or diffpy.structure.Atom
        If str the abbreviated name of the element.
        If int the atomic number of element.
    q: float or array-like
        Scattering vector, defined as q = (k' - k).
        In units of 1/Angstrom.
    use_xray: bool
        If True the xray atomic form factors are used.
        These values are converted to the electron scattering case by
        the Mott-Bethe formula.

    Returns
    -------
    fe(s): float
        The electron atomic scattering factor.

    """

    # s = (k' - k) / (4 * pi) = q / (4 * np.pi)
    s = np.asarray(q, dtype=dtype) / (4.0 * np.pi)

    # get element name as str
    element = get_element_name(element)

    if use_xray:
        # get correct row
        row = ATOMIC_SCATTERING_FACTORS_XRAY.loc[
            ATOMIC_SCATTERING_FACTORS_XRAY["Element"] == element
        ]

        out = (
            np.stack(
                tuple(
                    row[f"a{i}"].values * np.exp(-row[f"b{i}"].values * s**2)
                    for i in range(1, 5)
                ),  # 4 gaussians labelled 1-4
                axis=0,
            ).sum(axis=0)
            + row["c"].values
        )

        out = mott_bethe_approximation(out, atomic_numbers[element], s)
    else:
        # get correct row
        row = ATOMIC_SCATTERING_FACTORS.loc[
            ATOMIC_SCATTERING_FACTORS["Element"] == element
        ]

        out = np.stack(
            tuple(
                row[f"a{i}"].values * np.exp(-row[f"b{i}"].values * s**2)
                for i in range(1, 5)
            ),  # 4 gaussians labelled 1-4
            axis=0,
        ).sum(axis=0)

    return np.squeeze(out)


def reciprocal_vectors(a, b, *vectors):
    """
    Produce a* and b*, the reciprocal vectors of a and b.

    Parameters
    ----------
    a, b, *vectors: array-like
        Principal vectors.

    Returns
    -------
    star: np.ndarray
        Reciprocal vectors as np.array((a*, b*)).

    """
    return np.linalg.inv(
        (
            a,
            b,
            *vectors,
        )
    ).T


def generate_hkl_points(h=None, k=None, l=None, n=5):
    """
    Generate a set of hkl points for a reciprocal lattice.

    Parameters
    ----------
    h, k, l: None or array-like
        If None, then a set of points spanning [-n..n] will be used.
    n: int
        Default size [-n..n].

    Returns
    -------
    hkl: (N, 3) ndarray
        Set of hkl points.

    """

    if h is None:
        h = np.arange(-n, n + 1)
    if k is None:
        k = np.arange(-n, n + 1)
    if l is None:
        l = np.arange(-n, n + 1)

    grid = np.meshgrid(h, k, l)

    return np.column_stack(tuple(g.ravel() for g in grid))


def calculate_structure_factor(
    structure,
    g,
    scale_by_scattering_angle=True,
    debye_waller=True,
    T=293.0,
):
    """

    Calculates the structure factor for a set of hkl points.

    Parameters
    ----------
    structure: diffpy.structure.Structure
        Structure with a lattice and atoms.
    g: ([P,] N, 3) ndarray
        Scattering vectors in units 1/Angstrom.
    scale_by_scattering_angle: bool
        If True then the atomic scattering factor is computed as a
        function of scattering vector. If False then no scaling is
        applied as a function of scattering vector.
    debye_waller: bool
        If True then the Debye-Waller factor is used to compute the
        atomic scattering factor. The Debye-Waller factor is computed as
        :math:`exp(-Bs**2)`.
    T: float
        Temperature in Kelvin, used to calculate the Debye-Waller
        factor. Default is 293 K (room temperature).

    Returns
    -------
    F: (N,) ndarray
        Structure factor for each hkl point.

    """

    positions = get_positions(structure)

    # g vector magnitude
    g_abs = np.linalg.norm(g, axis=-1)

    # atomic scattering factors
    if scale_by_scattering_angle:
        f = np.transpose(
            [calculate_scattering_factor(atom, g_abs) for atom in structure]
        )
    else:
        # evaluate every atoms for 0 scattering vector
        f = np.transpose([calculate_scattering_factor(atom, 0.0) for atom in structure])

    if debye_waller:
        if False:  # isinstance(atoms, aseAtoms):
            # automatic filtering of BCC and FCC lattices with spacegroup
            # numbers 229 and 225
            if atoms.info["spacegroup"].no == 225:
                structure = "f.c.c."  # as written in table
            elif atoms.info["spacegroup"].no == 229:
                structure = "b.c.c."
            else:
                structure = None
        # TODO: spacegroup information for diffpy.structure
        else:
            structure = None
        B = np.array(
            [
                calculate_debye_waller_factor(atom, T, structure=structure)
                for atom in structure
            ]
        )
        # Debye-Waller factor is applied as exp(-Bs**2)
        f = f * np.exp(-1.0 * np.outer((g_abs / (4.0 * np.pi)) ** 2, B))

    # perform computation
    F = np.sum(f * np.exp(2.0 * np.pi * 1j * np.dot(g, positions.T)), axis=-1)
    return F


def calculate_reflection_intensity(F):
    """
    Return the intensity of the reflection which is calculated as:
    I = F * F*

    Parameters
    ----------
    F: ndarray, complex
        Structure factor.

    Returns
    -------
    I: ndarray, float
        Intensity.

    """

    return np.real(F * np.conjugate(F))


def calculate_ewald_sphere_radius(wavelength, dtype=DTYPE):
    """
    Convenience function to return radius of Ewald sphere.

    Parameters
    ----------
    wavelength: float
        Electron wavelength in Angstroms.

    Returns
    -------
    radius: float
        Radius of Ewald sphere in 1/Angstroms.

    """
    # Ewald sphere radius is 1/lambda
    return dtype(1.0 / wavelength)


def calculate_ewald_sphere_center(wavelength, psi=0.0, omega=0.0, dtype=DTYPE):
    """
    Calculate the center location of the Ewald sphere given a tilt and
    azimuthal angle.

    Parameters
    ----------
    wavelength: float
        Electron wavelength in Angstroms.
    psi: float
        The tilt angle of the incoming beam in radians.
    omega: float or (N,) ndarray
        The azimuthal angle of the incoming beam in radians.

    Returns
    -------
    center: (3,) ndarray
        The Ewald sphere center in units of 1/Angstroms.

    Notes
    -----
    [1] DOI: 10.1016/j.ultramic.2006.04.032

    """

    # sort out dtypes
    radius = calculate_ewald_sphere_radius(wavelength, dtype=dtype)
    omega = dtype(omega)

    if isinstance(omega, (list, tuple, np.ndarray)):
        psi = np.full_like(omega, psi, dtype=dtype)

    Kxy, Kz = (radius * np.sin(psi), radius * np.cos(psi))
    Kx, Ky = (Kxy * np.cos(omega), Kxy * np.sin(omega))

    return np.array((Kx, Ky, Kz))


def calculate_g_vectors(hkl, reciprocal_vectors):
    """
    Calculate g vectors from hkl Miller indices and the reciprocal
    lattice vectors.

    Parameters
    ----------
    hkl: (N, 3) ndarray
        Miller indices to compute.
    reciprocal_vectors: (3, 3) ndarray
        The reciprocal lattice vectors.

    Returns
    -------
    g: (N, 3) ndarray
        g vectors.

    """

    return np.dot(hkl, reciprocal_vectors)


def calculate_g_vector_scattering_angle(g, wavelength, degrees=True):
    """
    Return the angle between the projected in plane g(x, y) vector and
    the electron beam (z).

    Parameters
    ----------
    g: (N, 3) ndarray
        g vectors.
    wavelength: float
        Electron beam wavelength. In units of 1/g (typically Angstroms).
    degrees: boolean
        If True the angles are returned in degrees.

    Returns
    -------
    angles: (N,) ndarray
        The absolute angles between the projected g vectors and the
        electron beam.
    """
    angles = np.abs(
        np.arctan2(
            np.linalg.norm(g[..., :-1], axis=-1),
            calculate_ewald_sphere_radius(wavelength),
        )
    )

    return np.rad2deg(angles) if degrees else angles


def calculate_mosaicity_profile(g, s, mu, s0=0.0):
    """
    Calculate the effect of beam convergence and grain misorientation on
    diffraction intensities. The effect of these behaviours is modelled
    as a Gaussian to be convoluted with any rocking curve.

    Parameters
    ----------
    g: (N, 3) ndarray
        The g vectors in units of 1/Angstrom.
    s: (M,) ndarray
        The excitation error in units of 1/Angstrom.
    mu: float
        The mosaicity parameter in degrees.
    s0: float
        Mean excitation error.

    Returns
    -------
    gauss: (N, M) ndarray
        The Gaussian to convolve.

    Notes
    -----
    [1] DOI: 10.1107/S2052520619007534

    """
    sigma = np.deg2rad(mu) * np.linalg.norm(g, axis=1)
    M = (
        1.0
        / ((2.0 * np.pi) ** 0.5 * sigma)
        * np.exp(-(1.0 / 2.0) * np.square((s - s0) / sigma))
    )

    return M


def theta_to_k(theta, wavelength):
    """Convert scattering angle in radians to inverse units.

    Parameters
    ----------
    theta: array-like
        Scattering angle in radians.
    wavelength: float
        Electron wavelength.

    Returns
    -------
    k: array-like
        Scattering vector in inverse units. Units are the inverse of
        wavelength, ie. if wavelength is in Angstrom then k is in units
        of 1/Angstrom.

    Notes
    -----
    theta = r* * lambda
    """
    return theta / wavelength


def k_to_theta(k, wavelength):
    """Convert scattering angle in inverse units to radians.

    Parameters
    ----------
    k: array-like
        Scattering angle in inverse units.
    wavelength: float
        Electron wavelength in same units as k.

    Returns
    -------
    theta: array-like
        Scattering vector in radians.

    Notes
    -----
    theta = r* * lambda
    """
    return k * wavelength
