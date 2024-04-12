from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
from ipywidgets import Checkbox, IntSlider, interactive
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, DTypeLike, NDArray
from orix.quaternion import Orientation

from .rotation import generate_rotated_template
from .simulation import apply_point_spread_function
from .utils import DTYPE, _add_floats_to_array_2d, get_image_center


@dataclass
class DiffractionPattern:
    """
    Class containing diffraction pattern simulation data and parameters.
    Attribute .image accesses the simulated diffraction pattern.
    """

    image: NDArray
    shape: Tuple
    pixel_size: float
    center: ArrayLike
    psf: float
    orientation: Orientation
    direct_beam: bool
    center_of_mass_coordinates: bool
    scale_disks: bool

    @property
    def dtype(self) -> DTypeLike:
        return self.image.dtype

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.shape}"

    def plot(
        self,
        ax: Optional[Axes] = None,
        unit: Literal["data", "pixels"] = "data",
        **kwargs,
    ) -> None:
        """Plot diffraction pattern on axes."""
        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("cmap", "inferno")

        if unit.lower() == "data":
            ymin, xmin = tuple(-(0.5 + c) * self.pixel_size for c in self.center)
            ymax, xmax = tuple(
                (s - 0.5 - c) * self.pixel_size for s, c in zip(self.shape, self.center)
            )
            extent = (xmin, xmax, ymin, ymax)
            ax.set_xlabel("$\AA$")
            ax.set_ylabel("$\AA$")
        elif unit.lower() == "pixels":
            extent = None
        else:
            raise ValueError(f"unit must be one of 'data' or 'pixels'")

        ax.matshow(self.image, extent=extent, **kwargs)

    @classmethod
    def generate_diffraction_pattern(
        cls,
        g: ArrayLike,
        intensity: ArrayLike,
        shape: Tuple,
        orientation: Orientation,
        pixel_size: float,
        center: Union[ArrayLike, None] = None,
        psf: Union[float, ArrayLike] = 0,
        direct_beam: bool = True,
        center_of_mass_coordinates: bool = False,
        scale_disks: bool = False,
        dtype: DTypeLike = DTYPE,
    ) -> DiffractionPattern:
        """
        Create a mask of expected pixel positions from g vectors.
        Generated template is stored at self.template.

        Parameters
        ----------
        g: (N, 2) ndarray
            The simulated diffraction template coordinates (xy). In
            units of 1/Angstrom.
        shape: tuple
            Output mask size.
        pixel_size: float
            Pixel size in 1/Angstrom.
        center: None or tuple
            If None the center of shape will be used.
            Otherwise the center coordinates (xy) of the pattern.
        rotation: float
            Rotation to apply to the template in radians.
        psf: len(shape) ndarray or float
            Point spread function. All reflections are convoluted with
            this function. If ndarray then it is a psf kernel. If int or
            float then the template will be convoluted with a disk with
            this radius. Pass 0 for no psf to be applied.
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
        flipud: bool
            If True the projection is flipped vertically, in which case
            it matches ASTAR.

        Returns
        -------
        template: shape ndarray
            The simulated diffration template.

        """
        # init output array
        out = np.zeros(shape, dtype=dtype)

        if not direct_beam:
            mask_remove_000 = np.logical_not(np.isclose(g, 0).all(axis=1))
            g = g[mask_remove_000]
            intensity = intensity[mask_remove_000]

        # center the mask before rounding
        center = get_image_center(shape, center)

        # rotate, scale, and recenter coordinates
        # do not need z dimension as this is a projection
        # 0 azimuthal rotation is applied
        g = generate_rotated_template(
            g[..., : out.ndim][..., ::-1], 0, pixel_size, center
        )[
            ..., ::-1
        ]  # rotation is done in ij to be consistent and then swapped back to xy

        if not center_of_mass_coordinates:
            # center g then round to get pixel coordinates
            g = np.round(g).astype(int)

        # filter spots before adding to mask to a) save time and
        # b) prevent out of bounds IndexErrors
        mask_within_shape = np.logical_and(
            np.all(g >= 0, axis=1), np.all(g < shape, axis=1)
        )
        g = g[mask_within_shape]
        # also filter Is
        intensity = intensity[mask_within_shape]

        if scale_disks:
            # calculate after removing direct beam
            intensity_max = intensity.max()

            grid = np.stack(
                np.meshgrid(*(np.arange(i) for i in shape), indexing="xy"), axis=-1
            )

            for i, gi in enumerate(g):
                out[
                    np.linalg.norm(grid - gi, axis=-1)
                    <= psf * (intensity[i] / intensity_max)
                ] = intensity[i]
        else:
            if center_of_mass_coordinates:
                # updates array in place
                _add_floats_to_array_2d(
                    out, g[..., ::-1], intensity
                )  # [::-1] to convert from xy to ij
            else:
                # avoid possibilty of multiply indexing same point in
                # array, in this case the last index wins out and
                # determines the final value
                np.add.at(
                    out, tuple(g.T)[::-1], intensity
                )  # [::-1] to convert from xy to ij

            out = apply_point_spread_function(out, psf, mode="same")

        # create pattern
        diffraction_pattern = DiffractionPattern(
            image=out,
            shape=shape,
            pixel_size=pixel_size,
            center=center,
            orientation=orientation,
            psf=psf,
            direct_beam=direct_beam,
            center_of_mass_coordinates=center_of_mass_coordinates,
            scale_disks=scale_disks,
        )

        return diffraction_pattern


@dataclass
class DiffractionPatternBlock:
    """
    Class containing multiple 2d diffraction pattern simulation data
    and associated parameters. Attribute .data accesses the simulated
    diffraction patterns.
    """

    data: NDArray
    pixel_size: float
    center: Optional[ArrayLike]
    orientations: Orientation
    psf: float
    direct_beam: bool
    center_of_mass_coordinates: bool
    scale_disks: bool

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape[:-2]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def pattern_shape(self) -> Tuple[int, int]:
        return self.data.shape[-2:]

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @property
    def dtype(self) -> DTypeLike:
        return self.data.dtype

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.shape}"

    def __getitem__(self, indices) -> NDArray:
        return self.data[indices]

    @property
    def A(self) -> NDArray:
        """Generate A matrix for matrix decomposition."""
        return self.data.reshape(-1, np.prod(self.pattern_shape)).T

    def plot(self, ax: Optional[Axes] = None, cmap: str = "inferno") -> interactive:
        """Interactive plot."""
        if ax is None:
            fig, ax = plt.subplots()

        if self.ndim == 1:
            out = self._plot1d(ax, cmap)
        elif self.ndim == 2:
            out = self._plot2d(ax, cmap)
        elif self.ndim == 3:
            out = self._plot3d(ax, cmap)
        else:
            raise NotImplementedError(
                "Plotting for ndim > 3 is not currently implemented."
            )

        return out

    def _plot1d(self, ax: Axes, cmap: str) -> interactive:
        """Backend ndim=1 interactive plot."""
        ai = ax.matshow(self.data[0], cmap=cmap)

        def update(i, clim):
            ai.set_array(self.data[i])
            if clim:
                ai.set_clim(self.data[i].min(), self.data[i].max())

        return interactive(
            update,
            i=IntSlider(0, 0, self.shape[0] - 1),
            clim=Checkbox(True, description="Auto clim?"),
        )

    def _plot2d(self, ax: Axes, cmap: str) -> interactive:
        """Backend ndim=2 interactive plot."""
        ai = ax.matshow(self.data[0, 0], cmap=cmap)

        def update(i, j, clim):
            ai.set_array(self.data[i, j])
            if clim:
                ai.set_clim(self.data[i, j].min(), self.data[i, j].max())

        return interactive(
            update,
            i=IntSlider(0, 0, self.shape[0] - 1),
            j=IntSlider(0, 0, self.shape[1] - 1),
            clim=Checkbox(True, description="Auto clim?"),
        )

    def _plot3d(self, ax: Axes, cmap: str) -> interactive:
        """Backend ndim=3 interactive plot."""
        ai = ax.matshow(self.data[0, 0, 0], cmap=cmap)

        def update(i, j, k, clim):
            ai.set_array(self.data[i, j, k])
            if clim:
                ai.set_clim(self.data[i, j, k].min(), self.data[i, j, k].max())

        return interactive(
            update,
            i=IntSlider(0, 0, self.shape[0] - 1),
            j=IntSlider(0, 0, self.shape[1] - 1),
            k=IntSlider(0, 0, self.shape[2] - 1),
            clim=Checkbox(True, description="Auto clim?"),
        )
