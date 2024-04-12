from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional, Tuple, Union

import numpy as np
from diffpy.structure import Structure
from numpy.typing import ArrayLike, DTypeLike, NDArray
from orix.crystal_map import Phase
from orix.quaternion import Orientation, Rotation
from orix.vector import AxAngle, Vector3d
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from .microscope import electron_wavelength
from .orientations import rotation_between_vectors
from .reciprocal_lattice import (
    S_MAX,
    calculate_ewald_sphere_radius,
    calculate_g_vectors,
    calculate_reflection_intensity,
    calculate_structure_factor,
    generate_hkl_points,
    reciprocal_vectors,
)
from .sampling import generate_supersampled_grid
from .structure import get_unit_vectors, parse_structure
from .template import (
    DiffractionTemplate,
    DiffractionTemplateBlock,
    DiffractionTemplateBlockSuperSampled,
    DiffractionTemplateExcitationErrorModel,
    DiffractionTemplateExcitationErrorNorm,
)
from .utils import DTYPE


class DiffractionGeneratorType(str, Enum):
    CRYSTAL = "crystal"
    ATOMIC = "atomic"


@dataclass
class DiffractionGenerator(abc.ABC):
    """
    Generic electron diffraction generator.

    Parameters
    ----------
    structure: diffpy.structure.Structure
        Atomic structure including atomic positions and lattice.
    voltage: float
        Accelerating voltage in kV.
    max_angle: float
        The maxmium scattering angle to compute in degrees.
    atomic_scattering_factor: bool
        Whether to scale the scattering intensity with scattering angle.
    debye_waller: bool
        Whether to apply the Debye-Waller factor when calculating the
        scattering intensity.
    normalize: bool
        Whether to normalise base reflection intensities as a function
        of the direct beam value.
    n: int
        Number of ±hkl to compute.
    remove_direct_beam: bool
        If True hkl (0, 0, 0) is removed from the templates.
    dtype: dtype
        Datatype for calculations.
    """

    structure: Union[str, Path, Phase, Structure]
    voltage: float
    max_angle: float = 5  # degrees
    atomic_scattering_factor: bool = False
    debye_waller: bool = False
    normalize: bool = False
    n: int = 15
    remove_direct_beam: bool = True
    dtype: DTypeLike = DTYPE
    kind: ClassVar[DiffractionGeneratorType]

    @property
    def wavelength(self):
        return electron_wavelength(self.voltage)

    @property
    def hkl(self) -> NDArray:
        return self._hkl

    @hkl.setter
    def hkl(self, x: ArrayLike) -> None:
        x = np.asarray(x)
        if x.shape[-1] != 3:
            raise ValueError(f"x.shape[-1] must be 3, shape is {x.shape}.")
        self._hkl = x

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
    def g(self) -> NDArray:
        return self._g

    @g.setter
    def g(self, x: ArrayLike) -> None:
        x = np.asarray(x)
        if x.shape[-1] != 3:
            raise ValueError(f"x.shape[-1] must be 3, shape is {x.shape}.")
        self._g = x

    @property
    def reciprocal_vectors(self) -> NDArray:
        lattice_vectors = get_unit_vectors(self.structure)
        return reciprocal_vectors(*lattice_vectors)

    def __post_init__(self) -> None:
        if self.kind is not DiffractionGeneratorType.CRYSTAL:
            raise ValueError(f"{self.kind} kind is currently not supported.")
        self.structure = parse_structure(self.structure)
        # determine any abbreviations
        # do initial computation to filter reflections
        self.calculate_valid_reflections()

    def is_crystal(self) -> bool:
        """Returns True if if modelling crystal (hkl) diffraction."""
        return self.kind is DiffractionGeneratorType.CRYSTAL

    def filter_reflections_by_scattering_angle(self) -> None:
        """Calculate maximum scattering angle for all scattering vectors
        and filter those that are greater than max_angle."""
        mask = np.arctan(  # do not care about sign, only value
            np.linalg.norm(self.g, axis=-1)
            / calculate_ewald_sphere_radius(self.wavelength)
        ) <= np.deg2rad(self.max_angle)

        self._mask_data(mask)

    def _mask_data(self, mask: NDArray[np.bool_]) -> None:
        """Apply mask (of valid indices) to data arrays that are
        relevant to template generation."""
        self.g = self.g[mask]
        self.hkl = self.hkl[mask]
        self.structure_factor = self.structure_factor[mask]
        self.reflection_intensity = self.reflection_intensity[mask]

    def calculate_valid_reflections(self) -> None:
        """
        Filter forbidden reflections from all arrays.
        Limit to g vectors which scatter within max_angle.
        Calculate valid reflections.
        """
        self.hkl = generate_hkl_points(n=self.n)
        self.g = calculate_g_vectors(self.hkl, self.reciprocal_vectors)
        # calculate structure factor and scattering intensity
        # base reflection intensity is only scattering vector dependent
        # and not rotation dependent
        self.structure_factor = calculate_structure_factor(
            self.structure,
            self.g,
            scale_by_scattering_angle=self.atomic_scattering_factor,
            debye_waller=self.debye_waller,
        )
        self.reflection_intensity = calculate_reflection_intensity(
            self.structure_factor
        )
        mask_direct_beam = np.isclose(self.g, 0).all(axis=-1)
        mask_reflections = np.logical_not(mask_direct_beam)

        if self.normalize:
            # normalize reflection intensities as a fraction of direct beam intensity
            self.reflection_intensity[mask_reflections] /= self.reflection_intensity[
                mask_direct_beam
            ]

        if self.remove_direct_beam:
            self._mask_data(mask_reflections)

        # remove forbidden reflections which have 0 intensity
        self._mask_data(~np.isclose(self.reflection_intensity, 0))
        # limit to reflections within scattering angle less than max_angle
        self.filter_reflections_by_scattering_angle()

    def remove_duplicate_reflections(
        self, other: DiffractionGenerator, tolerance: float = 1e-2
    ) -> None:
        """Remove the set of reflections that are duplicated between
        this instance and another from this instance.

        Parameters
        ----------
        other
        tolerance
            Reflections in this instance less than or equal to this
            value from reflections in the other instance will be
            removed. This value is in units of 1/Angstrom.
        """
        # compute distance matrix between sets of g vectors
        dist = cdist(self.g, other.g)
        mask = (dist <= tolerance).any(axis=1)
        self._mask_data(~mask)

    def remove_twinned_reflections(self, misorientation: Rotation) -> None:
        """Remove twinned reflections from the set of valid Bragg
        reflections between misoriented crystals. For FCC metals Σ3
        twins are related by 60° rotation around <111> axis. The set of
        valid Bragg reflections are updated in place.

        Parameters
        ----------
        misorientation
            The misorientation which transforms one crystal orientation
            to the other. For Σ3 twinned crystals, this may be computed
            as `Rotation.from_axes_angles((1, -1, 1), np.deg2rad(60))`,
            for example.
        """
        if misorientation.size != 1:
            raise ValueError("Misorientation must be a single rotation.")

        g = Vector3d(self.g)
        g_rotated = ~misorientation * g

        # compute distance matrix between sets of g vectors
        dist = cdist(g.data, g_rotated.data)
        # overlapping points between the 2 sets will have 0 distance
        mask = np.isclose(dist, 0).any(axis=1)
        # remove them
        self._mask_data(~mask)

    def _remove_twinned_reflections_old(self, axis: ArrayLike) -> None:
        """Remove twinned reflections from the set of valid Bragg
        reflections between misoriented crystals. For FCC metals Σ3
        twins are related by 60° rotation around <111> axis. The set of
        valid Bragg reflections are updated in place.

        Parameters
        ----------
        axis
            The misorientation axis for the Σ3 twinned crystals, this
            must be a <111> axis.
        """
        v1 = np.array(axis).ravel()
        if (
            v1.size != 3
            or not np.allclose(np.abs(v1), np.abs(v1[0]))
            or not np.allclose(np.linalg.norm(axis), 3**0.5)
        ):
            raise ValueError("twin_axis should be a single <111> vector.")
        rot = rotation_between_vectors((0, 0, 1), v1)
        # get unit vectors from template
        g = self.g
        hkl = self.hkl
        norms = np.linalg.norm(hkl, axis=-1)
        # get in-twin-plane unit vectors, closest to direct beam
        mask = np.isclose(norms, 8**0.5)  # the twinned spots have hkl <220>
        g = g[mask]
        hkl = hkl[mask]
        norms = norms[mask]
        # of these vectors, they must be perpendicular to the tilt axis
        # ie. in the twin plane
        dot_products = hkl.dot(v1)
        mask = np.isclose(dot_products, 0)
        g = g[mask]
        hkl = hkl[mask]
        norms = norms[mask]
        # generate rotated g as in template
        g = (~rot * Vector3d(g)).data
        # choose vector most parallel to x as v2
        v2 = hkl[g.dot((1, 0, 0)).argmax()]
        # choose vector most perpendicular to v2 as v3
        cross_products = np.linalg.norm(np.cross(v2, hkl), axis=-1)
        v3 = hkl[cross_products.argmax()]
        # v2 and v3 are in the twin plane, v1 is perpendicular to these
        # and is the twin axis. Any hkl that can be made as a
        # combination of these vectors are shared and are removed
        v123 = np.array((v1.data, v2, v3))
        soln, *_ = np.linalg.lstsq(v123.T, self.hkl.T, rcond=None)
        # get the soln which are the same after rounding, ie. integer
        # solutions
        mask = np.isclose(soln.T, np.round(soln.T)).all(axis=-1)
        self._mask_data(np.logical_not(mask))

    def generate_templates(
        self,
        orientations: Rotation,
        s_max: float = S_MAX,
        psi: float = 0.0,
        omega: Union[float, ArrayLike] = 0.0,
        model: DiffractionTemplateExcitationErrorModel = DiffractionTemplateExcitationErrorModel.LINEAR,
        norm: DiffractionTemplateExcitationErrorNorm = DiffractionTemplateExcitationErrorNorm.NORM,
        flip: bool = True,
        dtype: DTypeLike = DTYPE,
    ) -> Union[DiffractionTemplate, DiffractionTemplateBlock]:
        """
        Simulate diffraction and generate resulting template.

        Parameters
        ----------
        orientations: orix.quaternion.Rotation
            Orientations of the crystal lattice from which diffraction
            is simulated.
        shape: None or tuple of ints
            If more than one rotation and shape is defined then this
            sets the shape of the output. If None then len(rotations) is
            used.
        s_max: float
            Maximum excitation error for a reflection to be excited.
            In 1/Angstrom.
        psi: float
            Precession angle.
        omega: (N,) float
            Template will be averaged over these precession phases.
            Typically (0, 2*np.pi) in radians.
        model: DiffractionTemplateExcitationErrorModel
            Eg. LINEAR or LORENTZIAN.
        norm: DiffractionTemplateExcitationErrorNorm
            Eg. NORM, REFERENCE.
        flip: bool
            If True then y coordinates are flipped to match ASTAR.
        dtype: DTypeLike
            Template datattype.

        Returns
        -------
        Either:
        template: DiffractionTemplate
            The simulated template if ony one template required.
        """
        if not isinstance(orientations, Rotation):
            raise ValueError("Orientations must be orix.quaternion.Rotation.")
        if not orientations.size >= 1:
            raise ValueError(
                f"Must be at least one orientation: size is {orientations.size}."
            )

        shape = orientations.shape
        # create holder for templates
        temp = np.empty(shape, dtype=object)

        for ijk in np.ndindex(shape):
            temp[ijk] = DiffractionTemplate.generate_template(
                structure=self.structure,
                g=self.g,
                hkl=self.hkl.view(),
                wavelength=self.wavelength,
                orientation=orientations[ijk],
                intensity=self.reflection_intensity,
                structure_factor=self.structure_factor,
                psi=psi,
                omega=omega,
                s_max=s_max,
                max_angle=self.max_angle,
                model=model,
                atomic_scattering_factor=self.atomic_scattering_factor,
                debye_waller=self.debye_waller,
                norm=norm,
                flip=flip,
                dtype=dtype,
            )

        if orientations.size == 1:
            out = temp.ravel()[0]  # return bare template
        else:  # more than one rotation
            out = DiffractionTemplateBlock(
                temp,
                wavelength=self.wavelength,
                s_max=s_max,
                psi=psi,
                omega=omega,
                model=model,
                norm=norm,
                flipped=flip,
                max_angle=self.max_angle,
                atomic_scattering_factor=self.atomic_scattering_factor,
                debye_waller=self.debye_waller,
            )
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} "
            + f"{self.structure.composition}, "
            + f"Voltage={self.voltage} kV"
            + ")"
        )

    def __str__(self) -> str:
        return (
            f"{repr(self)[:-1]}, "
            + f" Max. angle={self.max_angle} degrees, "
            + f" ASF={self.atomic_scattering_factor}, "
            + f" DW={self.debye_waller}"
            + ")"
        )

    def refine_template(
        self,
        template: DiffractionTemplate,
        image: NDArray,
        pixel_size: float,
        center: ArrayLike,
        num: int = 360,
        return_correlation_indices: bool = False,
        max_misorientation: Optional[float] = None,
        **kwargs,
    ) -> Union[DiffractionTemplate, Tuple[float, float]]:
        """
        Refine a template by maximizing its Correlation Index locally.

        Parameters
        ----------
        template: DiffractionTemplate
            Template to refine.
        image: ndarray
            Diffraction Pattern to match.
        pixel_size: float
            The pixel size in 1/Angstroms.
        center: ArrayLike
            The center coordinates of the diffraction pattern (ij).
        num: int
            The number of azimuthal rotations to compute per template.
        return_correlation_indices: bool
            If True a tuple of (CI_initial, CI_refined) is also returned.
        max_misorientation: scalar or None
            The refinement is constrained such that the misorientation is less than this value (degrees).
            If provided method should be either 'SLSQP' or 'COBYLA'.
            By default 'COBYLA' minimization procedure will be used.
        kwargs:
            Passed to scipy.optmize.minimize.

        Returns
        -------
        template: DiffractionTemplate
            The refined template.
        (CI_initial, CI_refined): tuple of floats
            The calculated Correlation Indices of the initial and refined templates.
            Onlt returned if return_correlation_indices is True.

        """
        center = np.asarray(center)

        def err(
            x: NDArray,
            template: DiffractionTemplate,
            image: NDArray,
            pixel_size: NDArray,
            center: NDArray,
            num: int,
        ):
            temp = self.generate_templates(
                Orientation.from_neo_euler(AxAngle(x)),
                template.s_max,
                template.psi,
                template.omega,
                template.model,
                template.norm,
                template.flipped,
                dtype=np.float64,  # solver requires 64-bit precision to get started
            )

            _, ci = temp.calculate_optimum_rotation(
                image,
                pixel_size,
                center,
                num=num,
                return_correlation_indices=True,
                norm_T=True,
                float_coords=True,
            )
            # maximize CI is to minimize negative
            return -1.0 * ci.max()

        if max_misorientation is not None:
            kwargs.setdefault("method", "COBYLA")

            def limit_misorientation(x: NDArray, init: NDArray, limit: float):
                o1 = Orientation.from_neo_euler(AxAngle(x))
                o2 = Orientation.from_neo_euler(AxAngle(init))
                misori = o2 * ~o1
                return limit - misori.angle.data

            constraint = dict(
                type="ineq",
                fun=limit_misorientation,
                args=(
                    AxAngle.from_rotation(template.orientation).data.ravel(),
                    np.deg2rad(max_misorientation),
                ),
            )

            kwargs.setdefault("constraints", []).append(constraint)
        else:
            kwargs.setdefault("method", "Nelder-Mead")

        # perform optimization
        rv = AxAngle.from_rotation(template.orientation).data.ravel()
        if rv.size != 3:
            raise ValueError("Rotation vector must have 3 components.")
        res = minimize(
            err,
            rv,
            args=(template, image, pixel_size, center, num),
            **kwargs,
        )
        # best orientation
        ori = Orientation.from_neo_euler(AxAngle(res.x))

        # generate refined template, scan azimuthal rotation and then combine the two
        temp = self.generate_templates(
            ori,
            template.s_max,
            template.psi,
            template.omega,
            template.model,
            template.norm,
            template.flipped,
        )

        r1, ci1 = temp.calculate_optimum_rotation(
            image,
            pixel_size,
            center,
            num=2 * num,  # extra azimuthal refinement
            return_correlation_indices=True,
            norm_T=True,
            float_coords=True,
        )

        rotation_azimuthal = Orientation.from_neo_euler(AxAngle((0, 0, -1.0 * r1)))

        # generate a new template including azimuthal rotation
        out = self.generate_templates(
            rotation_azimuthal * ori,
            template.s_max,
            template.psi,
            template.omega,
            template.model,
            template.norm,
            template.flipped,
        )

        r0, ci0 = template.calculate_optimum_rotation(
            image,
            pixel_size,
            center,
            num=2 * num,  # extra azimuthal refinement
            return_correlation_indices=True,
            norm_T=True,
            float_coords=True,
        )
        # scan optimum template rotation (again) to check we get the same maxmimum value
        # ci1_old, ci1_old_angle = ci1.max(), temp.generate_thetas(len(ci1))[ci1.argmax()]
        # ci1 = out.calculate_optimum_template_rotation(
        #     image,
        #     pixel_size,
        #     center,
        #     num=2 * num,  # extra azimuthal refinement
        #     return_all=True,
        #     norm_T=True,
        #     float_coords=True,
        # )

        if ci0.max() > ci1.max():
            print(
                "Optimization error occurred. Initial template has higher correlation "
                + f"index than refined template: {ci0.max():6f} > {ci1.max():6f}."
            )

        return (
            (
                out,
                (ci0.max(), ci1.max()),
            )
            if return_correlation_indices
            else out
        )

    def generate_template_block(
        self,
        grid: Optional[ArrayLike] = None,
        xrange: Optional[ArrayLike] = None,
        yrange: Optional[ArrayLike] = None,
        zrange: Optional[ArrayLike] = None,
        num: int = None,
        supersampling: int = 1,
        s_max: float = S_MAX,
        psi: float = 0.0,
        omega: Union[float, ArrayLike] = 0.0,
        model: DiffractionTemplateExcitationErrorModel = DiffractionTemplateExcitationErrorModel.LINEAR,
        norm: DiffractionTemplateExcitationErrorNorm = DiffractionTemplateExcitationErrorNorm.NORM,
        flip: bool = True,
        dtype: DTypeLike = DTYPE,
        progressbar: bool = True,
    ) -> Union[DiffractionTemplateBlock, DiffractionTemplateBlockSuperSampled]:
        """
        Generate a supersampled diffraction template grid.
        Each grid large grid point contains supersampling finer grid
        points.

        Parameters
        ----------
        grid: None or array-like
            If None then all other parameters must be defined.
            If provided then must be of the form (N, N, N, 3, s, s, s).
            If provided all other parameters are ignored.
        xrange, yrange, zrange: array-like
            (min, max) ranges for the main grid.
        num: int
            Number of sampling points for the main grid.
        supersampling: int
            Supersampling grid points are distributed between evenly
            around each main grid point. If equal to 1 then a normal
            grid is produced, ie. no supersampling.
        s_max: float
            Maximum excitation error for an excited reflection.
            In 1/Angstrom.
        psi: float
            Precession angle.
        omega: (N,) float
            Template will be averaged over these precession phases.
            Typically (0, 2*np.pi) in radians.
        model: DiffractionTemplateExcitationErrorModel
            Eg. LINEAR or LORENTZIAN.
        norm: DiffractionTemplateExcitationErrorNorm
            Eg. NORM, REFERENCE.
        flip: bool
            If True then y coordinates are flipped to match ASTAR.
        dtype: DTypeLike
            The datatype for the templates.
        progressbar: bool
            Whether to display the progressbar.

        Returns
        -------
        DiffractionTemplateBlockSuperSampled:
            The supersampled template grid.

        """
        # make sure either grid or other paramters needed to make the
        # grid are fully defined
        if grid is not None:
            assert (
                isinstance(grid, np.ndarray) and grid.ndim == 7
            ), "grid must be ndarray with ndim=7 and shape: (N, N, N, 3, s, s, s)."
        else:
            assert all(
                i is not None for i in (xrange, yrange, zrange, num, supersampling)
            ), "If grid is not provided then all other parameters must be defined."
            grid = generate_supersampled_grid(
                xrange, yrange, zrange, num, supersampling, dtype
            )

        ndim = 3  # possibly to fix later, eg 2d grid
        out = np.empty(grid.shape[:ndim], dtype=object)

        # now we have the grid...
        for ijk in tqdm(
            np.ndindex(grid.shape[:ndim]),
            total=np.prod(grid.shape[:ndim]),
            desc="Generating TemplateBlock",
            disable=not progressbar,
        ):
            sub_grid = np.stack(tuple(g.ravel() for g in grid[ijk]), axis=1)
            # produce templateblock
            out[ijk] = self.generate_templates(
                Rotation.from_neo_euler(AxAngle(sub_grid)),
                shape=grid.shape[-ndim:],
                s_max=s_max,
                psi=psi,
                omega=omega,
                model=model,
                norm=norm,
                flip=flip,
            )

        if supersampling > 1:
            out = DiffractionTemplateBlockSuperSampled(
                out,
                xrange,
                yrange,
                zrange,
                num,
                supersampling,
                wavelength=self.wavelength,
                s_max=s_max,
                psi=psi,
                omega=omega,
                model=model,
                norm=norm,
                flipped=flip,
                max_angle=self.max_angle,
                atomic_scattering_factor=self.atomic_scattering_factor,
                debye_waller=self.debye_waller,
                dtype=dtype,
            )
        else:
            out = DiffractionTemplateBlock(
                out,
                wavelength=self.wavelength,
                s_max=s_max,
                psi=psi,
                omega=omega,
                model=model,
                norm=norm,
                flipped=flip,
                max_angle=self.max_angle,
                atomic_scattering_factor=self.atomic_scattering_factor,
                debye_waller=self.debye_waller,
            )

        return out

    def generate_template_block_rotvecs(
        self,
        v1: Union[Rotation, ArrayLike] = (0, 1, 1),
        v2: Union[Rotation, ArrayLike] = (1, 1, 1),
        num: int = 1000,
        triangle: bool = True,
        s_max: float = S_MAX,
        psi: float = 0.0,
        omega: Union[float, ArrayLike] = 0,
        model: DiffractionTemplateExcitationErrorModel = DiffractionTemplateExcitationErrorModel.LINEAR,
        norm: DiffractionTemplateExcitationErrorNorm = DiffractionTemplateExcitationErrorNorm.NORM,
        flip: bool = True,
        dtype: DTypeLike = DTYPE,
    ) -> DiffractionTemplateBlock:
        """
        Generate a template block.
        Templates are generated in a grid along two directions.
        Direction1: from (0, 0, 1) to v1.
        Direction2: from v1 to v2.

        Parameters
        ----------
        v1: (3,) array-like or Rotation
            The first direction.
        v2: (3,) array-like or Rotation
            The second direction.
        num: int
            The number of points along each dimension.
        triangle: bool
            If True only a standard triangle is kept.
        s_max: float
            Maximum excitation error for a reflection to be excited.
            In 1/Angstrom.
        psi: float
            Precession angle.
        omega: (N,) float
            Template will be averaged over these precession phases.
            Typically (0, 2*np.pi) in radians.
        model: DiffractionTemplateExcitationErrorModel
            Eg. LINEAR or LORENTZIAN.
        norm: DiffractionTemplateExcitationErrorNorm
            Eg. NORM, REFERENCE.
        flip: bool
            If True then y coordinates are flipped to match ASTAR.
        dtype: DTypeLike
            Default Data Type.

        Returns
        -------
        DiffractionTemplateBlock
            The simulated template block.

        """
        if isinstance(v1, Rotation) and isinstance(v2, Rotation):
            # need angles
            a1 = v1.angle
            a2 = v2.angle
            # and unit vectors
            r1 = v1.axis / a1
            r2 = v2.axis / a2
        else:
            z = np.array((0, 0, 1), dtype=dtype)
            v1 = np.asarray(v1, dtype=dtype)
            v2 = np.asarray(v2, dtype=dtype)

            # normalise vectors
            z /= np.linalg.norm(z)
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)

            # find the rotation vector that transform z to v1, and v1 to v2
            r1 = np.cross(z, v1)
            r2 = np.cross(v1, v2)
            # normalise these vectors too
            r1 /= np.linalg.norm(r1)
            r2 /= np.linalg.norm(r2)

            # work out angles between the two vectors
            a1 = np.arccos(np.dot(z, v1))
            a2 = np.arccos(np.dot(v1, v2))

        # approximate side length of grid for n points
        side = int(((2 if triangle else 1) * num) ** 0.5 + 1)
        grid = np.meshgrid(
            np.linspace(0, a1, int(side)), np.linspace(0, a2, int(side)), indexing="xy"
        )

        # just keep lower traingle
        if triangle:
            idx = np.tril_indices(side)
        else:
            idx = np.ones_like(grid[0], dtype=bool)

        rot1 = Rotation.from_neo_euler(AxAngle(grid[0][idx][..., np.newaxis] * r1))
        rot2 = Rotation.from_neo_euler(AxAngle(grid[1][idx][..., np.newaxis] * r2))

        # combine the rotations, order matters
        rot = rot2 * rot1

        shape = (len(rot),) if triangle else (side, side)

        return self.generate_templates(
            rot,
            shape=shape,
            s_max=s_max,
            psi=psi,
            omega=omega,
            model=model,
            norm=norm,
            flip=flip,
        )


class CrystalDiffractionGenerator(DiffractionGenerator):
    """ """.join(("Crystal", *DiffractionGenerator.__doc__.split(" ")[1:]))

    kind = DiffractionGeneratorType.CRYSTAL

    def __init__(
        self,
        structure: Union[str, Path, Structure],
        voltage: float,
        max_angle: float = 5,  # degrees
        atomic_scattering_factor: bool = False,
        debye_waller: bool = False,
        normalize: bool = False,
        n: int = 15,
        remove_direct_beam: bool = True,
        dtype: DTypeLike = DTYPE,
    ):
        super().__init__(
            structure,
            voltage,
            max_angle=max_angle,
            atomic_scattering_factor=atomic_scattering_factor,
            debye_waller=debye_waller,
            normalize=normalize,
            n=n,
            remove_direct_beam=remove_direct_beam,
            dtype=dtype,
        )


class AtomicDiffractionGenerator(DiffractionGenerator):
    """ """.join(("Atomic", *DiffractionGenerator.__doc__.split(" ")[1:]))

    kind = DiffractionGeneratorType.ATOMIC

    def __init__(
        self,
        structure: Union[str, Path, Structure],
        voltage: float,
        max_angle: float = 5,  # degrees
        atomic_scattering_factor: bool = False,
        debye_waller: bool = False,
        normalize: bool = False,
        n: int = 15,
        remove_direct_beam: bool = True,
        dtype: DTypeLike = DTYPE,
    ):
        raise NotImplementedError("Atomic diffraction has not been implemented yet.")
        super().__init__(
            structure,
            voltage,
            max_angle=max_angle,
            atomic_scattering_factor=atomic_scattering_factor,
            debye_waller=debye_waller,
            normalize=normalize,
            n=n,
            remove_direct_beam=remove_direct_beam,
            dtype=dtype,
        )
