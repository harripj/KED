import numpy as np
from orix.quaternion.symmetry import Oh
from orix.vector import Vector3d

from ked.utils import get_orientations_standard_triangle


def test_get_orientations_standard_triangle():
    ori = get_orientations_standard_triangle(res=1)
    assert 1000 <= ori.size < 1200
    v1, v2 = np.array((0, 0, 1)), np.array((1, 1, 1))
    max_angle = np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    assert np.all(ori.angle <= max_angle)
    assert np.all((ori * Vector3d.zvector()) <= Oh.fundamental_sector)
