import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .utils import DTYPE


def generate_grid(
    xrange: ArrayLike,
    yrange: ArrayLike,
    zrange: ArrayLike,
    num: int,
    endpoint: bool = True,
    ravel: bool = True,
) -> np.ndarray:
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
    ravel: bool
        If True the grid is flattened to an (N**3, 3) array.

    Returns
    -------
    samples: (3, N, N, N) or (N**3, 3) ndarray
        The sampled values.

    """
    x = np.linspace(*xrange, num, endpoint=endpoint)
    y = np.linspace(*yrange, num, endpoint=endpoint)
    z = np.linspace(*zrange, num, endpoint=endpoint)

    grid = np.meshgrid(x, y, z, indexing="ij")

    if ravel:
        out = np.column_stack(tuple(g.ravel() for g in grid))
    else:
        out = np.stack(grid, axis=0)

    return out


def generate_supersampled_grid(
    xrange: ArrayLike,
    yrange: ArrayLike,
    zrange: ArrayLike,
    num: int,
    supersampling: int = 5,
    dtype: DTypeLike = DTYPE,
) -> np.ndarray:
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
    dtype: DTypeLike
        Dtype for the output grid.

    Returns
    -------
    samples: (num, num, num, 3, ss, ss, ss) ndarray
        The supersampled values.

    """
    if supersampling < 1:
        raise ValueError("Supersampling must be >= 1.")

    xspacing = (xrange[1] - xrange[0]) / num
    yspacing = (yrange[1] - yrange[0]) / num
    zspacing = (zrange[1] - zrange[0]) / num

    large_grid = generate_grid(xrange, yrange, zrange, num, endpoint=True, ravel=False)

    out = np.empty(
        large_grid.shape[1:] + (3, supersampling, supersampling, supersampling),
        dtype=dtype,
    )

    for i, j, k in np.ndindex(large_grid.shape[1:]):
        # grid center
        center = large_grid[:, i, j, k]
        # generate subgrid centered on a large_grid point
        sub_grid = generate_grid(
            (
                center[0] + (xspacing / 2) * (1 / supersampling - 1),
                center[0] + (xspacing / 2) * (1 / supersampling + 1),
            ),
            (
                center[1] + (yspacing / 2) * (1 / supersampling - 1),
                center[1] + (yspacing / 2) * (1 / supersampling + 1),
            ),
            (
                center[2] + (zspacing / 2) * (1 / supersampling - 1),
                center[2] + (zspacing / 2) * (1 / supersampling + 1),
            ),
            num=supersampling,
            endpoint=False,
            ravel=False,
        )

        out[i, j, k] = sub_grid

    return out
