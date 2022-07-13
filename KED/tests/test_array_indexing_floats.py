import numpy as np
import pytest
from scipy import ndimage
from skimage import measure

from KED.utils import add_floats_to_array, index_array_with_floats


@pytest.fixture
def size():
    return (128, 128)


@pytest.fixture
def coords():
    # coords were previously generated randomly within 128 array without
    # overlap
    return np.array(
        [
            [22.16592919, 84.2168154],
            [114.47476294, 64.45731195],
            [91.91697047, 26.66631115],
            [118.8173779, 103.96593876],
            [81.44861606, 13.51490295],
            [42.23061586, 48.05492847],
            [107.51921287, 126.35820592],
            [36.43436053, 43.07031768],
            [87.09934983, 18.88721496],
            [67.71800073, 116.66660002],
            [121.20380893, 77.10861935],
            [48.72294543, 1.48899105],
            [29.92450603, 4.34496871],
            [1.91161256, 42.90676203],
            [54.00934336, 5.97161717],
            [93.20684688, 45.96073859],
            [84.08467716, 4.52048007],
            [25.82118209, 47.41806082],
            [84.32867177, 33.55853447],
            [35.92985459, 110.52930006],
            [2.41754838, 48.54527727],
            [34.68800924, 75.66156361],
            [43.97492247, 113.5967165],
            [94.81890988, 29.13048797],
            [112.92559178, 27.1393408],
        ]
    )


@pytest.fixture
def vals(coords):
    return np.arange(len(coords)) + 1


@pytest.fixture
def non_overlapping_array(size, coords, vals):
    arr = np.zeros(size)
    add_floats_to_array(arr, coords, vals)
    return arr


def test_add_floats_to_array2d(non_overlapping_array, coords, vals):
    arr = non_overlapping_array
    labelled = measure.label(arr > 0, connectivity=1)
    com = ndimage.center_of_mass(arr, labelled, np.arange(labelled.max()) + 1)
    assert np.allclose(np.sort(com, axis=0), np.sort(coords, axis=0))
    # this is currently an approximate test
    assert np.isclose(arr.sum(), vals.sum())


def test_add_and_index_even_float_distribution():
    arr = np.zeros((16, 16))
    coord1 = (3.5, 6.5)  # evenly distributed
    val = 12
    add_floats_to_array(arr, np.atleast_2d(coord1), [val])
    assert arr.sum() == 12
    # index coords within range of active pixels
    coords = np.random.random_sample((25, 2)) + tuple(int(i) for i in coord1)
    vals = index_array_with_floats(arr, coords)
    assert np.allclose(vals, val / 4)  # each pixel has val / 4
    coord2 = (3, 6)
    val2 = index_array_with_floats(arr, np.atleast_2d(coord2))
    assert np.allclose(val2, val / 4)
    # test that going away from floor(coord1) reduced calculated value
    for ij in (0, 1):
        coord3 = list(coord2)
        coord3[ij] -= 1e-6  # subtract small offset
        val3 = index_array_with_floats(arr, np.atleast_2d(coord3))
        assert val3 < val / 4


def test_add_and_index_single_pixel_distribution():
    arr = np.zeros((16, 16))
    coord1 = (3.0, 6.0)  # evenly distributed
    val = 12
    add_floats_to_array(arr, np.atleast_2d(coord1), [val])
    assert arr.sum() == 12
    assert arr.max() == 12
    assert np.count_nonzero(arr) == 1
    assert ndimage.center_of_mass(arr) == coord1
    # index coords half a pixel away in each dir from active pixel
    coord2 = tuple(i + 0.5 for i in coord1)
    val2 = index_array_with_floats(arr, np.atleast_2d(coord2))
    assert np.allclose(val2, val / 4)
    coord3 = list(coord1)
    coord3[0] -= 0.5
    val3 = index_array_with_floats(arr, np.atleast_2d(coord3))
    assert np.allclose(val3, val / 2)
