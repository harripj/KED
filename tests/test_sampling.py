import numpy as np
from orix.quaternion import Orientation
import pytest

from ked.sampling import (
    SuperSampledOrientationGrid,
    generate_grid,
    generate_supersampled_grid,
)


class TestSuperSampledOrientationGrid:

    @pytest.mark.parametrize("degrees", [True, False])
    def test_from_axes_angles(self, degrees):
        xrange = (-1, 1)
        yrange = (-1.1, 1.1)
        zrange = (-2, 1.5)
        num = 5
        supersampling = 3
        grid1 = generate_supersampled_grid(
            xrange,
            yrange,
            zrange,
            num,
            supersampling=supersampling,
            as_orientation=False,
        )
        grid2 = generate_supersampled_grid(
            xrange,
            yrange,
            zrange,
            num,
            supersampling=supersampling,
            as_orientation=True,
            degrees=degrees,
        )
        grid1 = SuperSampledOrientationGrid.from_axes_angles(grid1, degrees=degrees)
        assert isinstance(grid1, SuperSampledOrientationGrid)
        assert grid1.shape == grid2.shape
        for ijk in np.ndindex(grid1.shape):
            assert grid1[ijk].shape == grid2[ijk].shape
            assert isinstance(grid1[ijk], Orientation)
            assert isinstance(grid2[ijk], Orientation)
            assert np.allclose(grid1[ijk].angle, grid2[ijk].angle)
            assert np.allclose(grid1[ijk].axis.data, grid2[ijk].axis.data)


@pytest.mark.parametrize("ravel", [True, False])
def test_generate_grid(ravel):
    xrange = (-1, 1)
    yrange = (-1.1, 1.1)
    zrange = (-2, 1.5)
    num = 5
    grid = generate_grid(xrange, yrange, zrange, num, ravel=ravel)
    assert grid.shape == (num**3, 3) if ravel else (num, num, num, 3)
    if ravel:
        assert len(np.unique(grid, axis=0)) == len(grid)
    else:
        # x
        assert (grid[0, :, :, 0] == xrange[0]).all()
        assert (grid[-1, :, :, 0] == xrange[1]).all()
        # y
        assert (grid[:, 0, :, 1] == yrange[0]).all()
        assert (grid[:, -1, :, 1] == yrange[1]).all()
        # z
        assert (grid[:, :, 0, 2] == zrange[0]).all()
        assert (grid[:, :, -1, 2] == zrange[1]).all()


@pytest.mark.parametrize("supersampling", [3, 4, 5])
def test_generate_supersampled_grid(supersampling):
    xrange = (-1, 1)
    yrange = (-1.1, 1.1)
    zrange = (-2, 1.5)
    num = 5
    grid = generate_supersampled_grid(
        xrange, yrange, zrange, num, supersampling=supersampling, as_orientation=False
    )
    odd = supersampling % 2
    assert grid.shape == (num, num, num, supersampling, supersampling, supersampling, 3)
    assert isinstance(grid, np.ndarray)
    assert grid.dtype != object
    for i in (0, -1):
        # x
        assert (grid[i, :, :, : supersampling // 2, :, :, 0] < xrange[i]).all()
        assert (grid[i, :, :, supersampling // 2 + odd :, :, :, 0] > xrange[i]).all()
        # y
        assert (grid[:, i, :, :, : supersampling // 2, :, 1] < yrange[i]).all()
        assert (grid[:, i, :, :, supersampling // 2 + odd :, :, 1] > yrange[i]).all()
        # z
        assert (grid[:, :, i, :, :, : supersampling // 2, 2] < zrange[i]).all()
        assert (grid[:, :, i, :, :, supersampling // 2 + odd :, 2] > zrange[i]).all()
        if odd:
            assert (grid[i, :, :, supersampling // 2, :, :, 0] == xrange[i]).all()
            assert (grid[:, i, :, :, supersampling // 2, :, 1] == yrange[i]).all()
            assert (grid[:, :, i, :, :, supersampling // 2, 2] == zrange[i]).all()
        else:
            assert ~np.isclose(grid[i, :, :, :, :, :, 0], xrange[i]).any()
            assert ~np.isclose(grid[:, i, :, :, :, :, 1], yrange[i]).any()
            assert ~np.isclose(grid[:, :, i, :, :, :, 2], zrange[i]).any()


@pytest.mark.parametrize("degrees", [True, False])
def test_generate_supersampled_grid_orientation(degrees):
    xrange = (-1, 1)
    yrange = (-1.1, 1.1)
    zrange = (-2, 1.5)
    num = 5
    supersampling = 4

    grid = generate_supersampled_grid(
        xrange,
        yrange,
        zrange,
        num,
        supersampling=supersampling,
        as_orientation=True,
        degrees=degrees,
    )
    assert isinstance(grid, SuperSampledOrientationGrid)
    assert grid.shape == (num, num, num)
    assert grid.dtype == object
    assert isinstance(grid[0, 0, 0], Orientation)
    assert grid[0, 0, 0].shape == (supersampling, supersampling, supersampling)
    max_angle = (xrange[0] ** 2 + yrange[0] ** 2 + zrange[0] ** 2) ** 0.5
    max_grid_angle = grid[0, 0, 0][0, 0, 0].angle
    if degrees:
        max_grid_angle = np.rad2deg(max_grid_angle)
    else:
        max_angle = np.deg2rad(max_angle)
    assert max_grid_angle > max_angle
