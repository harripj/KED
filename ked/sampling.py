from typing import Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from orix.quaternion import Orientation

from .utils import DTYPE


class SuperSampledOrientationGrid(np.ndarray):
    def __new__(cls, arr: ArrayLike):
        obj = np.asarray(arr)
        if obj.dtype != object or obj.ndim != 3:
            raise ValueError(
                "The input array must be of object dtype and shape (N, N, N)"
            )
        return obj.view(cls)

    @classmethod
    def from_axes_angles(
        cls, grid: NDArray, degrees: bool = False
    ) -> "SuperSampledOrientationGrid":
        if grid.ndim != 7 or grid.shape[-1] != 3:
            raise ValueError(
                "The input array must be of shape (N, N, N, ss, ss, ss, 3)"
            )
        obj = np.empty(grid.shape[:3], dtype=object)
        for ijk in np.ndindex(obj.shape):
            obj[ijk] = Orientation.from_axes_angles(
                grid[ijk], np.linalg.norm(grid[ijk], axis=-1), degrees=degrees
            )
        return cls(obj)


def generate_grid(
    xrange: ArrayLike,
    yrange: ArrayLike,
    zrange: ArrayLike,
    num: int,
    endpoint: bool = True,
    as_orientation: bool = True,
    degrees: bool = False,
    ravel: bool = False,
) -> Union[NDArray, Orientation]:
    """
    Generate a grid with even sampling over a specified range.

    Parameters
    ----------
    xrange, yrange, zrange: array-like
        Pairs of (min, max) ranges.
    num: int
        The number of samples over each range.
    endpoint: bool
        Whether endpoint is included within the range.
    as_orientation
        If True then the created sub grids are treated as axes-angles
        orientations and are cast as `orix.quaternion.Orientation`.
    degrees
        If `as_orientation` is `True` then this flag is passed to treat
        the input grid as degrees.
    ravel: bool
        If True the grid is flattened to an (N**3, 3) array.

    Returns
    -------
    samples
        Return shape is (N, N, N, 3) or (N**3, 3) if ravel is True.
    """
    x = np.linspace(*xrange, num, endpoint=endpoint)
    y = np.linspace(*yrange, num, endpoint=endpoint)
    z = np.linspace(*zrange, num, endpoint=endpoint)

    grid = np.meshgrid(x, y, z, indexing="ij")

    if ravel:
        out = np.column_stack(tuple(g.ravel() for g in grid))
    else:
        out = np.stack(grid, axis=-1)

    if as_orientation:
        out = Orientation.from_axes_angles(
            out, np.linalg.norm(out, axis=-1), degrees=degrees
        )

    return out


def generate_supersampled_grid(
    xrange: ArrayLike,
    yrange: ArrayLike,
    zrange: ArrayLike,
    num: int,
    supersampling: int = 5,
    as_orientation: bool = True,
    degrees: bool = False,
    dtype: DTypeLike = DTYPE,
) -> Union[NDArray, SuperSampledOrientationGrid]:
    """
    Generate an evenly supersampled grid over a specified range.

    Parameters
    ----------
    xrange, yrange, zrange: array-like
        Pairs of (min, max) ranges for the main grid.
    num: int
        The number of samples over each range of the main grid.
    supersampling: int
        Subsampling factor of the fine grid.
    as_orientation
        If True then the created sub grids are treated as axes-angles
        orientations and are cast as `orix.quaternion.Orientation`.
    degrees
        If `as_orientation` is `True` then this flag is passed to treat
        the input grid as degrees.
    dtype: DTypeLike
        Data type for the output grid.
        If `as_orientation` is `True` then the return type is `object`

    Returns
    -------
    samples:
        (N, N, N) array of `orix.quaternion.Orientation` objects if
        `as_orientation` is `True`.
        (N, N, N, supersampling, supersampling, supersampling, 3) array
        otherwise.
    """
    if supersampling < 1:
        raise ValueError("Supersampling must be >= 1.")

    xmin, xmax = xrange
    ymin, ymax = yrange
    zmin, zmax = zrange

    xspacing = (xmax - xmin) / num
    yspacing = (ymax - ymin) / num
    zspacing = (zmax - zmin) / num

    large_grid = generate_grid(
        xrange, yrange, zrange, num, endpoint=True, as_orientation=False, ravel=False
    )

    out_shape = large_grid.shape[:-1]
    if not as_orientation:
        out_shape += (supersampling, supersampling, supersampling, 3)

    out = np.empty(out_shape, dtype=object if as_orientation else dtype)
    for ijk in np.ndindex(large_grid.shape[:-1]):
        # grid center
        cx, cy, cz = large_grid[ijk]
        # generate subgrid centered on a large_grid point
        out[ijk] = generate_grid(
            (
                cx + (xspacing / 2) * (1 / supersampling - 1),
                cx + (xspacing / 2) * (1 / supersampling + 1),
            ),
            (
                cy + (yspacing / 2) * (1 / supersampling - 1),
                cy + (yspacing / 2) * (1 / supersampling + 1),
            ),
            (
                cz + (zspacing / 2) * (1 / supersampling - 1),
                cz + (zspacing / 2) * (1 / supersampling + 1),
            ),
            num=supersampling,
            endpoint=False,
            as_orientation=as_orientation,
            degrees=degrees,
            ravel=False,
        )

    return SuperSampledOrientationGrid(out) if as_orientation else out
