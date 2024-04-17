from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import itertools
from typing import Callable, Generator, Optional, Tuple, Union

from diffpy.structure import Structure
from ipywidgets import IntSlider, interactive
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from orix.quaternion import Orientation, Quaternion
from orix.vector import Vector3d
from skimage import transform
from tqdm.auto import tqdm

from .indexing import (
    _calculate_correlation_index,
    _scan_azimuthal_rotations,
    _scan_pixel_sizes,
)
from .orientations import convert_to_orix
from .pattern import DiffractionPattern, DiffractionPatternBlock
from .process import virtual_reconstruction
from .reciprocal_lattice import (
    S_MAX,
    calculate_ewald_sphere_center,
    calculate_ewald_sphere_radius,
)
from .rotation import generate_rotated_template
from .simulation import apply_point_spread_function
from .structure import get_unit_cell_volume
from .utils import DTYPE, calculate_zone_axis, generate_thetas, get_image_center


class DiffractionTemplateExcitationErrorModel(Enum):
    LINEAR = 1
    LORENTZIAN = 2


class DiffractionTemplateExcitationErrorNorm(Enum):
    NORM = 1
    REFERENCE = 2
    LATTICE = 3


def _calculate_excitation_error_z_lattice(
    g: NDArray,
    ewald_sphere_center: ArrayLike,
    ewald_sphere_radius: float,
    z_lattice: ArrayLike,
) -> NDArray:
    """
    Calculate excitation error where relrods are extended along a certain direction.

    Parameters
    ----------
    g: (N, 3) ndarray
        Reciprocal lattice vectors.
    ewald_sphere_center: (3,) array-like
        The Ewald sphere center location in the same coordinate system.
    ewald_sphere_radius: float
        Radius of the Ewald sphere in 1/Angstrom.
    z_lattice: (3,) array-like
        Relrods are extended along this vector.
        The excitation error is therefore calculated along this direction.
        For thin specimens this is normally the z-direction, hence the naming.

    Returns
    -------
    s: (N,) ndarray
        The excitation error for each reciprocal lattice vector.

    """

    # force z unit vector
    z_lattice /= np.linalg.norm(z_lattice)

    # factors for quadratic equation
    # https://math.stackexchange.com/a/1939462
    a = np.dot(z_lattice, z_lattice)
    b = 2 * np.dot(z_lattice, (g - ewald_sphere_center).T)
    delta_g = g - ewald_sphere_center
    # diagonal dot product
    c = np.einsum("ij,ij->i", delta_g, delta_g) - ewald_sphere_radius**2

    # solve the quadratic
    pm = -1  # only want the lower solution, ie. around 000
    soln = (-b + pm * np.sqrt(b**2 - 4 * a * c)) / (2 * a)  # general quadratic solution

    # two root quadratic solution
    # soln = np.stack(tuple((- b + pm * np.sqrt(b ** 2 - 4*a*c)) / (2 * a) for pm in (-1, 1)))

    # as z is a unit vector, the solution is in units of z (Angtsroms)
    # and therefore |soln| is directly the |excitation error|
    # s is negative if g is outside the sphere
    sign = (
        -1
        if np.linalg.norm(ewald_sphere_center - z_lattice) < ewald_sphere_radius
        else 1
    )
    return soln * sign


def calculate_excitation_error(
    g: NDArray,
    wavelength: float,
    psi: float = 0.0,
    omega: float = 0.0,
    norm: DiffractionTemplateExcitationErrorNorm = DiffractionTemplateExcitationErrorNorm.NORM,
    z_lattice: Optional[ArrayLike] = None,
    dtype: DTypeLike = DTYPE,
) -> NDArray:
    """
    Calculate the excitation errors between the Ewald sphere and a set
    of hkl points.

    Parameters
    ----------
    g: (N, 3) ndarray
        Set of g vectors.
    wavelength: float
        Wavelength of electron beam in Angstroms.
    psi, omega: floats
        The tilt and azimuthal angles of the incoming beam in radians.
    precession: bool
        If True then the effect of full precession is considered.
        The excitation error is calculated for each azimuthal rotation
        angle of the precession.
    norm: DiffractionTemplateExcitationErrorModel
        If NORM then the geometric norm between the lattice point and
        the Ewald sphere center is used.
        If REFERENCE then the excitation error is calculated with along
        the reference frame z-axis.
        If LATTICE then the excitation error is calculated along the
        lattice z-axis. In the latter case z_lattice direction must also
        be defined.
    z_lattice: (3,) array-like
        The lattice z vector direction in the Cartesian reference frame.
        For no lattice rotation, this would be (0, 0, 1)

    Returns
    -------
    s: ((P,), N,) ndarray
        Excitation error for each hkl point.
        If precession is True then a value of s is calculated for each
        azimuthal rotation angle of the precession P.

    """
    if not isinstance(norm, DiffractionTemplateExcitationErrorNorm):
        raise TypeError("Norm must be of type DiffractionTemplateExcitationErrorNorm.")

    # format g input
    g = np.asarray(g, dtype=dtype)
    if g.ndim == 1:
        g = g[np.newaxis]
    assert g.ndim == 2 and g.shape[1] == 3, "g should be (3,) or (N, 3) ndarray."

    # define ewald sphere
    ewald_sphere_radius = calculate_ewald_sphere_radius(wavelength, dtype=dtype)
    # if omega has more than one value assume precession
    precession = isinstance(omega, (list, tuple, np.ndarray)) and len(omega) > 1

    if precession:
        ewald_sphere_center = calculate_ewald_sphere_center(
            wavelength, psi, omega, dtype=dtype
        )
        # calculate the sphere position z for every g vector around one full precession
        _sqrt = np.sqrt(
            ewald_sphere_radius**2
            - np.square(ewald_sphere_center.T[:, np.newaxis, :] - g)[..., :-1].sum(
                axis=-1
            )
        )
        z = np.stack(
            (
                ewald_sphere_center[-1][..., np.newaxis] + _sqrt,
                ewald_sphere_center[-1][..., np.newaxis] - _sqrt,
            ),
            axis=-1,
        ).min(axis=-1)
        # excitation error is then the difference between gz and z
        s = (g[..., -1] - z).T
    else:
        ewald_sphere_center = calculate_ewald_sphere_center(
            wavelength,
            psi=psi,
            omega=np.squeeze(omega),  # squeeze in case iterable of just one value
            dtype=dtype,
        )
        # calculate difference between each g-vector and the Ewald
        # sphere center. The excitation error is the difference between
        # this value and the sphere radius (1/lambda)
        if norm is DiffractionTemplateExcitationErrorNorm.NORM:
            s = (
                np.linalg.norm(g - ewald_sphere_center, axis=1) - ewald_sphere_radius
            )  # shortest vector
        elif norm is DiffractionTemplateExcitationErrorNorm.REFERENCE:
            s = _calculate_excitation_error_z_lattice(
                g, ewald_sphere_center, ewald_sphere_radius, z_lattice
            )

            ### OLD METHOD BELOW, same values as new method above
            # # take the minimum distance between the positive and negative sqrt
            # _sqrt = np.sqrt(
            #     ewald_sphere_radius ** 2
            #     - np.square(ewald_sphere_center - g)[..., :-1].sum(axis=-1)
            # )
            # # z coordinate of the Ewald sphere surface at each (x, y) g-vector position
            # z = np.stack(
            #     (ewald_sphere_center[-1] + _sqrt, ewald_sphere_center[-1] - _sqrt),
            #     axis=1,
            # ).min(axis=-1)
            # # excitation error is then the difference between gz and z
            # s = g[..., -1] - z

        elif norm is DiffractionTemplateExcitationErrorNorm.LATTICE:
            if z_lattice is None:
                raise ValueError("Lattice z_direction vector must be defined.")
            s = _calculate_excitation_error_z_lattice(
                g, ewald_sphere_center, ewald_sphere_radius, z_lattice
            )
        else:
            raise ValueError(f"{norm} is not in DiffractionTemplateExcitationErrorNorm")

    return s


def integrate_reflection_intensity(
    I: NDArray,
    s: NDArray,
    s_max: float = S_MAX,
    model: DiffractionTemplateExcitationErrorModel = DiffractionTemplateExcitationErrorModel.LINEAR,
    gamma: Optional[NDArray] = None,
    volume: Optional[float] = None,
    wavelength: Optional[float] = None,
    dtype: DTypeLike = DTYPE,
):
    """
    Calculate the expected reflection intensities.
    Each reflection has a bas intensity I, which decreases linearly with
    excitation error s.

    Parameters
    ----------
    I: (N,) ndarray
        Base Reflection intensities.
    s: ((P,), N,) ndarray
        Excitation errors.
        If s was calculated for multiple precession angles then s is 2D.
        In this case the average reflection intensity over the
        precession is returned.
    s_max: float
        The maximum excitation error, used in linear model.
    model: str
        Either 'linear' or 'lorentzian'.
    gamma: (N,) ndarray
        Structure factor for each reflection.
    volume: float
        Unit cell volume in Angstrom^3.
    wavelength: float
        Electron wavelength in Angstroms.

    Returns
    -------
    Is: (N,) ndarry
        The reflection intensity.

    Notes
    -----
    [1] DOI: 10.1107/S2052520619007534

    """

    I = dtype(I)
    s = dtype(s)

    if model is DiffractionTemplateExcitationErrorModel.LORENTZIAN:
        if any((i is None for i in (gamma, volume, wavelength))):
            raise ValueError(
                "gamma, volume, and wavelength must be defined for the Lorenztian model."
            )
        xi = dtype(
            np.pi * volume / (wavelength * np.abs(gamma))
        )  # gamma possibly imaginary, just take abs
        out = ((I / 2.0) * (1 / ((xi * s.T) ** 2 + 1))).T
    elif model is DiffractionTemplateExcitationErrorModel.LINEAR:
        ds = np.abs(s)  # we only care about magnitude of s, not sign
        ds[ds > s_max] = s_max  # no intensity if ds is greater than excitation error
        out = (
            I - (I / s_max) * ds.T
        ).T  # smaller excitation error gives larger intensity
    else:
        raise ValueError("Model is not in DiffractionTemplateExcitationErrorModel.")

    # s.ndim == 2 in precession, for example, in which case we average
    # over the rotation
    if s.ndim == 2:
        out = out.mean(axis=1, dtype=dtype)

    return out


@dataclass
class DiffractionTemplate:
    """
    Class containing template simulation data and parameters.
    Instance method generate_diffraction_pattern can be used to simulate
    a diffraction pattern.
    """

    structure: Structure
    g: ArrayLike
    hkl: ArrayLike
    wavelength: float  # in Angstroms
    orientation: Tuple
    intensity: ArrayLike
    excitation_error: ArrayLike
    base_intensity: ArrayLike
    s_max: float
    norm: DiffractionTemplateExcitationErrorNorm
    psi: float
    omega: Union[float, ArrayLike]
    model: DiffractionTemplateExcitationErrorModel
    max_angle: float
    atomic_scattering_factor: bool
    debye_waller: bool
    flipped: bool  # whether gy was flipped (to match ASTAR)
    dtype: DTypeLike = DTYPE

    @property
    def x_star(self) -> NDArray:
        return self.g[..., 0]

    @property
    def y_star(self) -> NDArray:
        return self.g[..., 1]

    @property
    def z_star(self) -> NDArray:
        return self.g[..., 2]

    @property
    def projected_vectors(self) -> NDArray:
        """Return the diffraction vectors projected onto the xy plane."""
        return self.g[..., :-1]

    @property
    def h(self) -> NDArray:
        return self.hkl[..., 0]

    @property
    def k(self) -> NDArray:
        return self.hkl[..., 1]

    @property
    def l(self) -> NDArray:
        return self.hkl[..., 2]

    @property
    def filter_all(self):
        return (
            self.filter_excitation_error
            * self.filter_forbidden_reflections
            * self.filter_scattering_angle
        )

    @property
    def direct_beam_mask(self) -> NDArray[np.bool_]:
        """Returns mask where every g vector except direct beam is True."""
        return np.logical_not(np.isclose(self.g, 0).all(axis=-1))

    @property
    def g_radius(self) -> NDArray:
        """The radial distance of the g vectors."""
        return np.linalg.norm(self.g[..., :-1], axis=-1)

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        pixel_size: float = 1.0,
        center: Tuple = (0, 0),
        labels: bool = False,
        rotation: float = 0.0,
        size: float = 10.0,
        plot_center: bool = True,
        scale_by_intensity: bool = True,
        flip_y: bool = False,
        kwargs_center: dict = dict(),
        kwargs_text: dict = dict(),
        **kwargs,
    ):
        """
        Plot excited reflection locations on an axes.

        Parameters
        ----------
        ax: plt.Axes or None
            Axes to plot on. If None then a new figure will be created.
        pixel_size: float
            Scale factor for plot.
            Defaults to 1, ie. data units (1/Angstrom).
        center: tuple
            Center coordinates (xy), in same units as data.
        labels: bool
            If True then the Template Miller indices are also plotted.
        rotation: float
            Template azimuthal rotation in radians.
        size: float
            The maximum size of the spots on the plot.
            The larger the circle, the more intense the reflection.
        plot_center: bool
            Plot direct beam.
        scale_by_intensity: bool
            If True then plotted circles are scaled by reflection
            intensity. Otherwise all circles will have the same size.
        coordinates: str
            Either 'xy' or 'ij'.
        kwargs_center: dict
            Kwargs for direct beam. Passed to plt.Axes.scatter.
        kwargs_text: dict
            Kwargs for labels. Passed to plt.Axes.text.
        kwargs:
            Passed to ax.plot.

        Returns
        -------
        ax: plt.Axes
            The axes (new or otherwise) that was plotted on.

        """

        kwargs.setdefault("color", "r")
        kwargs.setdefault("ls", "None")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("facecolor", (0,) * 4)

        db_mask = self.direct_beam_mask

        # get xy coordinates and convert to ij
        temp = self.projected_vectors[db_mask]
        # for calculations the rotation is performed in ij coords
        # => to keep the rotation consistent, rotate in ij and then
        # convert back to xy
        temp = generate_rotated_template(
            temp[..., ::-1], rotation, pixel_size, np.zeros(temp.shape[1])
        )[..., ::-1]
        if flip_y:
            temp[..., 1] *= -1
        temp += center

        # work out circle sizing, ignore direct beam
        if scale_by_intensity:
            s = size * self.intensity[db_mask] / self.intensity[db_mask].max()
        else:
            s = np.full((np.count_nonzero(db_mask),), size)

        # plot
        if ax is None:
            was_ax = False
            fig, ax = plt.subplots(figsize=plt.figaspect(1))
        else:
            was_ax = True

        ax.scatter(*temp.T, s=s, **kwargs)

        if plot_center:
            kwargs_center.setdefault("marker", "x")
            kwargs_center.setdefault("color", "w")
            kwargs_center.setdefault("s", size * 0.75)
            # add direct beam to plot
            ax.scatter(*center, **kwargs_center)  # plot direct beam

        if labels:
            kwargs_text.setdefault("size", 8)
            kwargs_text.setdefault("fontdict", dict(family="Arial"))
            for i, xy in enumerate(temp):
                ax.text(
                    *xy,
                    s=str(tuple(self.hkl[i])).replace(" ", ""),
                    ha="center",
                    va="bottom",
                    **kwargs_text,
                )

        if not was_ax:
            _sort_template_ax(ax, self)

        return ax

    def __repr__(self):
        o = convert_to_orix(self.orientation)
        return f"{self.__class__.__name__} {np.rad2deg(o.to_euler()).round(2)}"

    @property
    def zone_axis(self):
        """Return approximate Z zone axis and misorientation (in radians)."""
        return calculate_zone_axis(self.orientation)

    def generate_diffraction_pattern(
        self,
        shape: Tuple,
        pixel_size: float,
        center: Optional[ArrayLike] = None,
        psf: Union[float, ArrayLike] = 0.0,
        direct_beam: bool = False,
        center_of_mass_coordinates: bool = False,
        scale_disks: bool = False,
        dtype: DTypeLike = DTYPE,
    ) -> DiffractionPattern:
        """
        Generate diffraction pattern image from this template.
        See staticmethod of the same name for more information.

        Parameters
        ----------
        shape: tuple
            Output image shape.
        pixel_size: float
            Pixel size in 1/Angstrom.
        center: None or tuple
            If None the center of shape will be used.
        psf: len(shape) ndarray or float
            Point spread function. All reflections are convoluted with
            this function. If ndarray then it is a psf kernel.
            If int or float then the template will be convoluted with a
            disk with this radius. Pass 0 for no psf to be applied.
            This value is used as the maximum disk size if scale_disks
            is True.
        direct_beam: bool
            If False then (0, 0, 0) g vector is removed.
        center_of_mass: bool
            If True the reflection coordinates are added to the
            projection using their center of mass. If False the
            coordinates are rounded before being added.
        scale_disks: bool
            If True the diffraction disk radii are scaled with intensity
            and excitation error.
        dtype: dtype or bool
            Output datatype for mask. Useful types are bool and float.

        Returns
        -------
        pattern: DiffractionPattern
            The simulated pattern.
            Also accessible at self.diffraction_pattern.

        """
        self.diffraction_pattern = DiffractionPattern.generate_diffraction_pattern(
            g=self.g,
            intensity=self.intensity,
            shape=shape,
            pixel_size=pixel_size,
            center=center,
            orientation=self.orientation,
            psf=psf,
            direct_beam=direct_beam,
            center_of_mass_coordinates=center_of_mass_coordinates,
            scale_disks=scale_disks,
            dtype=dtype,
        )
        return self.diffraction_pattern

    def calculate_optimum_rotation(
        self,
        image: NDArray,
        pixel_size: float,
        center: Optional[ArrayLike] = None,
        num: int = 360,
        float_coords: bool = False,
        norm_P: bool = False,
        norm_T: bool = False,
        return_correlation_indices: bool = False,
    ) -> Union[float, Tuple[float, NDArray]]:
        """
        Calculate the optimum rotation of this template for a given
        pattern.

        Parameters
        ----------
        image: (N, M) ndarray
            The experimental diffraction pattern to index.
        pixel_size: float
            The pixel sizes of the image in 1/Angstrom.
        center: (2,) array-like or None
            The direct beam coodinates of the image (ij).
            If None then is the pattern center is assumed to be the
            center of the image.
        num: int
            The number of rotation steps in the range [0..2*pi] over
            which to compute the optimum rotation.
        float_coords: bool
            If True then distribute vectors within array by center of
            mass (more accurate, but takes longer). Otherwise round
            vectors to index image.
        norm_P, norm_T: bool
            Whether to normalize the Pattern and Template intensities.
        return_correlation_indices: bool
            If True then the Correlation Index array is also returned
            for each angle.

        Returns
        -------
        angle: float
            The optimum template rotation angle in radians.
        correlation_index: ndarray, optional
            The calculated Correlation Index score for all queried
            angles. Returned if return_correlation_indices is True.
        """
        image = np.asarray(image)
        assert image.ndim == 2, "image must be 2d array."

        center = get_image_center(image.shape, center)

        thetas = generate_thetas(num)
        result = np.empty(num, dtype=self.dtype)

        _scan_azimuthal_rotations(
            self.projected_vectors[..., ::-1],
            image,
            pixel_size,
            center,
            thetas,
            self.intensity,
            result,
            float_coords=float_coords,
            norm_P=norm_P,
            norm_T=norm_T,
        )
        rotation = thetas[result.argmax()]
        return (rotation, result) if return_correlation_indices else rotation

    def virtual_reconstruction(
        self,
        data: Union[NDArray, Generator[NDArray, None, None]],
        pixel_size: float,
        center: Optional[ArrayLike] = None,
        scale_intensity: bool = True,
        normP: bool = False,
        normT: bool = False,
        sum: bool = True,
        fn: Optional[Callable] = None,
    ) -> NDArray:
        """
        Compute a Frozen Template Virtual Reconstruction.

        Parameters
        ----------
        data: ndarray or generator
            Data may be 2d (individual frame) or more, eg. 4d.
            The last two axes are assumed to be the image axes.
        pixel_size: float
            Pixel size in 1/Angstrom.
        center: None or array-like
            Diffraction pattern center (ij). If None it is assumed to be
            the frame center.
        scale_intensity: bool
            If True the VR is calculated using the template intensities.
            Otherwise each reflection is worth the same amount (1).
        normP, normT: bool
            Normalize VR by Pattern and Template intensities
            respectively.
        fn: None or Callable
            Function to apply to each frame before reconstruction.
            If None the raw data is used.
            If provided it must be of the form x = fn(x).

        """
        if isinstance(data, np.ndarray):
            assert data.ndim >= 2, "data must be at least 2d."
            iterate = False
            is_generator = False
        elif isinstance(data, (list, tuple)):
            initial = data[0]
            assert initial.ndim == 2, "data must be at least 2d."
            iterate = True
            is_generator = False
        # assume generator. Itertools.chain does not pass GeneratorType
        # hence hasattr
        elif hasattr(data, "__iter__"):
            initial = next(data)
            assert (
                isinstance(initial, np.ndarray) and initial.ndim == 2
            ), "Each iteration must yield 2d ndarray."
            iterate = True
            is_generator = True
        else:
            raise ValueError("data is not ndarray nor iterable.")

        if iterate:
            center = get_image_center(initial.shape[-2:], center)
        else:
            center = get_image_center(data.shape[-2:], center)

        coords = self.projected_vectors_to_pixels(pixel_size, center)

        if is_generator:
            # put back initial frame before calculation
            data = itertools.chain((initial,), data)

        return virtual_reconstruction(
            data,
            coords,
            self.intensity if scale_intensity else None,
            fn=fn,
            normP=normP,
            normT=normT,
            sum=sum,
            dtype=self.dtype,
        )

    def projected_vectors_to_pixels(
        self, pixel_size: float, center: ArrayLike
    ) -> NDArray:
        """
        Convert projected vectors in inverse units to pixel coordinates.
        The (x*, y*) template coordinates are converted to detector
        (i, j) coordinates.

        Parameters
        ----------
        pixel_size: scalar
            Pixel size of the detector in 1/Angstrom.
        center: array-like
            Center of the detector in pixel coordinates (ij).

        Returns
        -------
        ij: (N, 2) ndarray
            The projected coordinates on the detector.

        """

        return self.projected_vectors[..., ::-1] / pixel_size + center

    def calculate_correlation_index_1d(
        self,
        radial_profile: NDArray,
        bins: Optional[NDArray] = None,
        pixel_size: Optional[float] = None,
        norm_P: bool = False,
        norm_T: bool = True,
        bin_centers: Optional[NDArray] = None,
    ) -> float:
        """Calculate correlation index from 1d radial profile.

        Parameters
        ----------
        radial_profile : (N,) array-like
            Radial line profile of the image.
        bins : array-like, optional
            The precalculated histogram bin edges, by default None.
            Either bins or pixel_size must be defined.
        pixel_size : float, optional
            The pixel size of the the data points in radial_profile in
            1/Angstrom, by default None. If pixel_size is not defined,
            bin_centers must be defined.
        norm_P, norm_T: bool
            Normalize the pattern and template values, respectively.
        bin_centers: array-like, optional
            The precomputed bin center indices for the radial_profile
            array. Must be defined if bins is defined.

        Returns
        -------
        CI: float
            The calculated 1d correlation index.
        """
        if bins is None and pixel_size is None:
            raise ValueError("One of bins or pixel_size must be defined.")

        if pixel_size is not None:
            bins = radial_profile.size
            range = (0, radial_profile.size * pixel_size)
        else:
            range = None
            if bin_centers is None:
                raise ValueError(
                    "bin_centers must be an integer array if bins is defined."
                )

        vals, bins = np.histogram(
            self.g_radius, bins=bins, range=range, weights=self.intensity
        )
        if bin_centers is None:
            bin_centers = (bins[:-1] / pixel_size).astype(int)

        return _calculate_correlation_index(
            radial_profile[bin_centers], vals, norm_P=norm_P, norm_T=norm_T
        )

    def scan_pixel_sizes(
        self,
        image: NDArray,
        pixel_sizes: ArrayLike,
        center: Optional[ArrayLike] = None,
        ax: Optional[plt.Axes] = None,
        num: int = 360,
        return_all: bool = False,
        float_coords: bool = False,
        dtype: DTypeLike = DTYPE,
    ) -> Union[Tuple[float, float], NDArray]:
        """
        Compute the best correlation index from a range of pixel sizes.
        The template correlation index is computed for each pixel sizes
        over a full range of rotations.

        Parameters
        ----------
        template: DiffractionTemplate
            The simulated diffraction template.
        image: (N, M) ndarray
            The experimental diffraction pattern to index.
        center: (2,) array-like
            The direct beam coodinates of the image (ij).
        pixel_sizes: array-like
            The pixel sizes to scan.
        ax: None or plt.Axes
            If provided then the pixel size scan results are plotted.
        num: int
            Number of rotation points to consider over range [0.. 2*pi].
        return_all: bool
            If True then full scan results are plotted.
            If False then tuple of optimum (pixel size, theta) is
            returned.
        float_coords: bool
            If True then distribute vectors within array by center of
            mass (more accurate, but takes longer). Otherwise round
            vectors to index image.
        dtype: dtype
            Output datatype.

        Returns
        -------
        Either:
        (CI, theta): (N, 2) ndarray
            The calculated correlation index for each pixel_size and
            optimum rotation value. If return_all is True, there is one
            result for every pixel size.
        (pixel size, theta): (2,) tuple
            If return_all is False then only the optimum pixel size and
            theta are returned.

        """
        center = get_image_center(image.shape, center)
        out = np.empty((len(pixel_sizes), 2), dtype=dtype)

        _scan_pixel_sizes(
            self.projected_vectors[..., ::-1],
            image,
            center,
            pixel_sizes,
            generate_thetas(num),
            self.intensity,
            out,
            float_coords=float_coords,
        )

        if ax:
            # plotCI result
            ax.plot(pixel_sizes, out[:, 0])

            # plot theta value
            color_theta = "tab:red"
            axtheta: Axes = ax.twinx()
            axtheta.plot(pixel_sizes, out[:, 1], color=color_theta, ls="dotted")

            # label axes
            ax.set_xlabel("Pixel size")
            ax.set_ylabel("CI")

            axtheta.set_ylabel(r"$\theta$", color=color_theta)
            axtheta.spines["right"].set_color(color_theta)
            axtheta.tick_params(axis="y", colors=color_theta)

        # return CI value along with corresponding optimum rotation for
        # each pixel size or only optimum
        if not return_all:
            i = out[:, 0].argmax()
            out = (pixel_sizes[i], out[i, 1])

        return out

    @classmethod
    def generate_template(
        cls,
        structure: Structure,
        g: ArrayLike,
        hkl: ArrayLike,
        wavelength: float,
        orientation: Quaternion,
        intensity: ArrayLike,
        structure_factor: ArrayLike,
        psi: float,
        omega: ArrayLike,
        s_max: float,
        max_angle: float,
        model: DiffractionTemplateExcitationErrorModel,
        atomic_scattering_factor: bool,
        debye_waller: bool,
        norm: str,
        flip: bool,
        dtype: DTypeLike = DTYPE,
    ) -> DiffractionTemplate:
        """
        Simulate kinematic diffraction and generate resulting template.

        Parameters
        ----------
        structure: diffpy.structure.Structure
            The structre of the unit cell including a lattice and atoms.
        g: (N, 3) ndarray
            The g vectors in 1/Angstrom (xyz).
        hkl: (N, 3) ndarray
            The associated hkl vectors.
        wavelength: float
            The electron wavelength in Angstroms.
        orientation: orix.quaternion.Quaternion
            Orientations of the crystal lattice from which to simulate
            diffraction.
        intensity: (N,) ndarray
            The base reflection intensity for each reflection.
        structure_factor: (N,) ndarray
            The structure factor for each reflection.
        psi: float
            Precession angle.
        omega: (N,) float
            Template will be averaged over these precession phases.
            Typically (0, 2*np.pi) in radians.
        s_max: float
            Maximum excitation error for a reflection to be excited.
            In units of 1/Angstrom
        max_angle: float
            The maximum reflection angle in degrees.
        model: DiffractionTemplateExcitationErrorModel
            Eg. LINEAR or LORENTZIAN.
        atomic_scattering_factor: bool
            If True then the scattering intensity scales with scattering
             angle.
            See atomic scattering factor.
        debye_waller: bool
            If True the Debye-Waller factor is applied.
        norm: DiffractionTemplateExcitationErrorNorm
            Eg. NORM or REFERENCE.
        flip: bool
            If True then y coordinates are flipped to match ASTAR.
        extra_factor: float
            Reflections within extra_factor * s_max will be kept for
            future refinement.

        Returns
        -------
        template(s): (N,) DiffractionTemplates
            The simulated template.

        """
        if not isinstance(orientation, Quaternion):
            raise TypeError("orientation must be orix.quaternion.Quaternion.")

        # orix defined rotations as passive rotations of the sample
        # reference frame, as is the Bunge definition
        # rotation of the reciprocal lattice is an active rotation of
        # the vectors in the lab reference frame, therefore this
        # rotation is inverted
        g_rotated = (~orientation * Vector3d(g)).data

        # calculate the excitation error between the reciprocal lattice
        # points and the Ewald sphere
        s = calculate_excitation_error(
            g_rotated, wavelength, psi=psi, omega=omega, norm=norm
        )

        volume = get_unit_cell_volume(structure)
        # the intensity will be integrated over the range of excitation
        # errors which will be more than 1 in the case of precession
        Is = integrate_reflection_intensity(
            intensity,
            s,
            s_max,
            model=model,
            gamma=structure_factor,
            volume=volume,
            wavelength=wavelength,
            dtype=dtype,
        )

        # filter by excitation error
        mask = np.abs(s) <= s_max

        if flip:
            g_rotated[..., 1] *= -1

        # create Template
        return cls(
            structure=structure,
            g=g_rotated[mask],
            hkl=hkl[mask],
            wavelength=wavelength,
            orientation=orientation,
            intensity=Is[mask],
            excitation_error=s[mask],
            base_intensity=intensity[mask],
            s_max=s_max,
            norm=norm,
            psi=psi,
            omega=omega,
            model=model,
            max_angle=max_angle,
            atomic_scattering_factor=atomic_scattering_factor,
            debye_waller=debye_waller,
            flipped=flip,
            dtype=dtype,
        )


@dataclass
class DiffractionTemplateBlock:
    templates: NDArray[np.object_]
    wavelength: float
    s_max: float
    norm: DiffractionTemplateExcitationErrorNorm
    psi: Union[float, ArrayLike]
    omega: float
    model: DiffractionTemplateExcitationErrorModel
    max_angle: float
    atomic_scattering_factor: bool
    debye_waller: bool
    flipped: bool
    dtype: DTypeLike = DTYPE

    def __getitem__(self, indices):
        return self.templates[indices]

    def __len__(self):
        return len(self.templates)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self.shape}"

    def ravel(self):
        return self.templates.ravel()

    def flatten(self):
        return self.ravel()

    @property
    def shape(self):
        return self.templates.shape

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def ndim(self):
        return self.templates.ndim

    def calculate_optimum_template(
        self,
        image: NDArray,
        pixel_size: float,
        center: Union[None, ArrayLike] = None,
        num: int = 360,
        float_coords: bool = False,
        return_correlation_indices: bool = False,
    ) -> Tuple[DiffractionTemplate, NDArray]:
        """Calculate the best template for a given pattern from within
        this DiffractionTemplateBlock.

        Parameters
        ----------
        image: ndarray
            The diffraction pattern to index.
        pixel_size: float
            The pixel size of the pattern in 1/Angstrom.
        center: array-like or None
            The (ij) center of the diffraction pattern in the image.
            If None then the center is assumed to be the center of the
            image.
        num: int
            The number of azimuthal rotations over the range [0..360] to
            test for each diffraction pattern.
        float_coords: bool
            If True then the float coordinates of the projected template
            reflections on the image are used.
        return_correlation_indices: bool
            If True the correlation indices for each template are also
            returned.

        Returns
        -------
        template: DiffractionTemplate
            The optimum template.
        CI: ndarray
            The correlation indices for each template. Returned if
            return_correlation_indices is True.
        """
        temp = self.ravel()
        result = np.empty(len(temp), dtype=self.dtype)
        # calculate correlation index for every template
        for i, template in enumerate(tqdm(temp)):
            template: DiffractionTemplate = template
            _, CI = template.calculate_optimum_rotation(
                image,
                pixel_size,
                center,
                num=num,
                float_coords=float_coords,
                norm_P=False,
                norm_T=True,
                return_correlation_indices=True,
            )
            result[i] = CI.max()
        optimum_template = temp[result.argmax()]
        return (
            (optimum_template, result)
            if return_correlation_indices
            else optimum_template
        )

    def calculate_optimum_template_1d(
        self,
        image: NDArray,
        pixel_size: float,
        full_match_n: int = 0,
        center: Optional[ArrayLike] = None,
        num: int = 360,
        norm_P: bool = False,
        norm_T: bool = True,
        float_coords: bool = False,
        progressbar: bool = False,
    ) -> float:
        """Calculate 1d correlation index on this template on a polar
        transformed diffraction pattern.

        Parameters
        ----------
        image: (M, N) ndarray
            Image on which to perform template matching process. This
            image will be polar transformed.
        pixel_size: float
            The pixel size of the image in 1/Angstrom.
        full_match_n : int
            After initial 1d matching the best n matches are selected
            for a full matching pass, default is 0.
        center : None or array-like
            The center of the diffraction pattern. If None then the
            center of the image will be used.
        num : int
            The number of azimuthal rotations to test for full
            matching. Only used if full_match_n is greater than 0.
        norm_P, norm_T: bool
            Normalize the pattern and template values, respectively.
        float_coords : bool
            Whether to use center of mass indices in indexing.
        progressbar : bool
            Show a progressbar during the matching process.

        Returns
        -------
        CI: float
            The calculated correlation index.
        """
        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError("image must be 2d.")
        # compute image polar transform
        polar_image = transform.warp_polar(image)
        # calculate 1d polar line profile
        radial_profile = polar_image.sum(axis=0)
        # flatten templates
        templates = self.ravel()
        CI = np.empty(len(templates), dtype=self.dtype)
        for i, template in enumerate(tqdm(templates, disable=not progressbar)):
            if not i:
                # define histogram bins only once, as they are defined by polar image
                bins = np.histogram_bin_edges(
                    template.g_radius,
                    bins=radial_profile.size,
                    range=(0, radial_profile.size * pixel_size),
                )
                # bin should be exactly integers, ignore right hand bin edge
                bin_centers = (bins[:-1] / pixel_size).astype(int)
            CI[i] = template.calculate_correlation_index_1d(
                radial_profile,
                bins=bins,
                bin_centers=bin_centers,
                norm_P=norm_P,
                norm_T=norm_T,
            )
            # # compute histogram
            # vals, _ = np.histogram(
            #     template.g_radius,
            #     bins=bins,
            #     weights=template.intensity,  # weight by reflection intensity
            # )
            # CI[i] = _calculate_correlation_index(
            #     polar_line_profile[bin_centers], vals, norm_P=norm_P, norm_T=norm_T
            # )
        if full_match_n:
            # get best n matches
            indices = np.argsort(CI)[::-1][:full_match_n]
            for i in indices:
                template = templates[i]
                _, CI_full = template.calculate_optimum_rotation(
                    image,
                    pixel_size,
                    center,
                    num=num,
                    float_coords=float_coords,
                    norm_P=False,
                    norm_T=True,
                    return_correlation_indices=True,
                )
                CI[i] = CI_full.max()
        return CI

    def plot(
        self,
        ax: Optional[Axes] = None,
        labels: bool = True,
        size: int = 10,
        facecolor: Optional[Union[Tuple[int, int, int, int], str]] = None,
    ):
        """Interactive plot."""
        if facecolor is None:
            facecolor = (0, 0, 0, 0)

        if ax is None:
            fig, ax = plt.subplots()

        p0 = ax.scatter(
            [],
            [],
            color="tab:red",
            facecolor=facecolor,
        )

        _sort_template_ax(ax, self.templates.ravel()[0])

        def update_plot(template):
            p0.set_offsets(template.projected_vectors)
            p0.set_sizes(template.intensity * size / template.intensity.max())

        if self.ndim == 1:

            def update(i):
                update_plot(self.templates[i])

            out = interactive(
                update,
                i=IntSlider(0, 0, self.shape[0] - 1),
            )

        elif self.ndim == 2:

            def update(i, j):
                update_plot(self.templates[i, j])

            out = interactive(
                update,
                i=IntSlider(0, 0, self.shape[0] - 1),
                j=IntSlider(0, 0, self.shape[1] - 1),
            )

        elif self.ndim == 3:

            def update(i, j, k):
                update_plot(self.templates[i, j, k])

            out = interactive(
                update,
                i=IntSlider(0, 0, self.shape[0] - 1),
                j=IntSlider(0, 0, self.shape[1] - 1),
                k=IntSlider(0, 0, self.shape[2] - 1),
            )

        else:
            raise ValueError(
                "Grid must be 1-3 dimensional. Other dimensionalities are currently unsupported."
            )

        return out

    def generate_diffraction_patterns(
        self,
        shape: ArrayLike,
        pixel_size: float,
        center: Optional[ArrayLike] = None,
        psf: Union[NDArray, float] = 0.0,
        direct_beam: bool = False,
        center_of_mass_coordinates: bool = True,
        scale_disks: bool = False,
        dtype: DTypeLike = DTYPE,
        progressbar: bool = True,
        keep_references: bool = True,
    ) -> DiffractionPatternBlock:
        """
        Generate a diffraction pattern image for every template in the
        block. See staticmethod of the same name for more information.

        Parameters
        ----------
        shape: tuple
            Output image shape.
        pixel_size: float
            Pixel size in 1/Angstrom.
        center: None or tuple
            If None the center of shape will be used.
        psf: len(shape) ndarray or float
            Point spread function. All reflections are convoluted with
            this function. If ndarray then it is a psf kernel.
            If int or float then the template will be convoluted with a
            disk with this radius. Pass 0 for no psf to be applied. This
            value is used as the maximum disk size if scale_disks is
            True.
        direct_beam: bool
            If False then (0, 0, 0) g vector is removed.
        center_of_mass: bool
            If True the reflection coordinates are added to the
            projection using their center of mass. If False the
            coordinates are rounded before being added.
        scale_disks: bool
            If True the diffraction disk radii are scaled with intensity
            and excitation error.
        dtype: dtype or bool
            Output datatype for mask. Useful types are bool and float.
        progressbar: bool
            Whether to show the progress bar.
        keep_reference: bool
            Whether a reference will kept to each template.
            Also whether a reference will be kept between the instance
            of this class and the DiffractionPatternBlock.

        Returns
        -------
        pattern: DiffractionPatternBlock
            The simulated patterns.
            Also accessible at self.templates.diffraction_pattern.

        """
        # create arrays for data
        data = np.empty(self.shape + tuple(shape), dtype=dtype)
        orientations = Orientation.identity(self.shape)
        # loop over each template and produce a pattern
        for ijk in tqdm(
            np.ndindex(self.shape),
            desc="Generating patterns",
            total=np.prod(self.shape),
            disable=not progressbar,
        ):
            template: DiffractionTemplate = self.templates[ijk]
            pattern = DiffractionPattern.generate_diffraction_pattern(
                g=template.projected_vectors,
                intensity=template.intensity,
                shape=shape,
                pixel_size=pixel_size,
                center=center,
                psf=psf,
                orientation=template.orientation,
                direct_beam=direct_beam,
                center_of_mass_coordinates=center_of_mass_coordinates,
                scale_disks=scale_disks,
                dtype=dtype,
            )
            data[ijk] = pattern.image
            orientations[ijk] = template.orientation

            if keep_references:
                template.diffraction_pattern = pattern

        pattern_block = DiffractionPatternBlock(
            data=data,
            pixel_size=pixel_size,
            center=center,
            orientations=orientations,
            psf=psf,
            direct_beam=direct_beam,
            center_of_mass_coordinates=center_of_mass_coordinates,
            scale_disks=scale_disks,
        )

        if keep_references:
            self.diffraction_patterns = pattern_block

        return pattern_block


@dataclass
class DiffractionTemplateBlockSuperSampled:
    """
    Class to hold diffraction templates from a supersampled (up to) 3D
    grid. Each grid point contains a DiffractionTemplateBlock.
    """

    templates: NDArray[np.object_]
    wavelength: float
    s_max: float
    norm: DiffractionTemplateExcitationErrorNorm
    psi: Union[float, ArrayLike]
    omega: float
    model: DiffractionTemplateExcitationErrorModel
    max_angle: float
    atomic_scattering_factor: bool
    debye_waller: bool
    flipped: bool
    dtype: DTypeLike = DTYPE

    @property
    def supersampling(self) -> Tuple[int, int, int]:
        return self.templates.ravel()[0].shape

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.shape}"

    @property
    def shape(self):
        return self.templates.shape

    @property
    def size(self):
        return self.templates.size

    @property
    def ndim(self):
        return self.templates.ndim

    def ravel(self):
        return self.templates.ravel()

    def flatten(self):
        return self.ravel()

    def __getitem__(self, indices) -> DiffractionTemplateBlock:
        return self.templates[indices]

    def generate_diffraction_patterns(
        self,
        shape: ArrayLike,
        pixel_size: float,
        center: Optional[ArrayLike] = None,
        psf: Union[NDArray, float] = 0.0,
        direct_beam: bool = False,
        center_of_mass_coordinates: bool = False,
        scale_disks: bool = False,
        dtype: DTypeLike = DTYPE,
        keep_references: bool = False,
    ) -> DiffractionPatternBlock:
        """
        Generate average diffraction pattern image for every
        supersampled grid in the block. See staticmethod of the same
        name for more information.

        Parameters
        ----------
        shape: tuple
            Output image shape.
        pixel_size: float
            Pixel size in 1/Angstrom.
        center: None or tuple
            If None the center of shape will be used.
        psf: len(shape) ndarray or float
            Point spread function. All reflections are convoluted with
            this function. If ndarray then it is a psf kernel. If int or
            float then the template will be convoluted with a disk with
            this radius. Pass 0 for no psf to be applied. This value is
            used as the maximum disk size if scale_disks is True.
        direct_beam: bool
            If False then (0, 0, 0) g vector is removed.
        center_of_mass: bool
            If True the reflection coordinates are added to the
            projection using their center of mass. If False the
            coordinates are rounded before being added.
        scale_disks: bool
            If True the diffraction disk radii are scaled with intensity
            and excitation error.
        dtype: DTypeLike
            The dtype of the patterns.
        keep_references: bool
            If True then the intermediate computed patterns are retained
            by the DiffractionTemplateBlocks. If False then the
            references are not kept and the intermediate data will be
            garbage collected. Set to False to keep memory usage down.
            If True memory usage will increase by supersampling**ndim.

        Returns
        -------
        pattern: DiffractionPatternBlock
            The simulated patterns averaged over the fine grid.

        """

        out = np.empty(self.shape + shape, dtype=dtype)

        for ijk in tqdm(
            np.ndindex(self.shape),
            total=self.size,
            desc="Generating averaged patterns",
        ):
            # each array element is a TemplateBlock
            # so generate the PatternBlock
            template: DiffractionTemplateBlock = self.templates[ijk]
            patterns: DiffractionPatternBlock = template.generate_diffraction_patterns(
                shape,
                pixel_size,
                center=center,
                psf=0,  # apply psf after averaging
                direct_beam=direct_beam,
                center_of_mass_coordinates=center_of_mass_coordinates,
                scale_disks=scale_disks,
                dtype=dtype,
                disable_tqdm=True,
                keep_references=keep_references,  # intermediate patterns
            )

            # average the patternblock
            average = patterns.data.mean(axis=tuple(range(patterns.ndim)))
            out[ijk] = apply_point_spread_function(average, psf)

        return DiffractionPatternBlock(
            data=out,
            pixel_size=pixel_size,
            center=center,
            psf=psf,
            direct_beam=direct_beam,
            center_of_mass_coordinates=center_of_mass_coordinates,
            scale_disks=scale_disks,
        )


def _sort_template_ax(ax: Axes, template: DiffractionTemplate) -> None:
    """Sort out axes for template plotting."""
    max_radius = (
        1.1  # padding factor for axes
        * np.tan(np.deg2rad(template.max_angle))
        * calculate_ewald_sphere_radius(template.wavelength)
    )
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    # Angstrom units
    ax.set_xlabel("$\AA$")
    ax.set_ylabel("$\AA$")

    ax.set_aspect("equal")
