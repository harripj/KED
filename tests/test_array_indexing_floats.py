from ked123.utils import add_floats_to_array, index_array_with_floats
import numpy as np
from scipy import ndimage
from skimage import measure


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
