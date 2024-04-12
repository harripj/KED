import numpy as np
from numpy.typing import NDArray
from orix.quaternion import Orientation
from orix.vector import Vector3d

from ked.reciprocal_lattice import reciprocal_vectors


def explicit_reciprocal_vectors(vectors: NDArray):
    """Standard definition of reciprocal lattice vectors.

    Notes
    -----
    [1] https://en.wikipedia.org/wiki/Reciprocal_lattice
    """
    vectors = np.asarray(vectors)
    if not vectors.shape == (3, 3):
        raise ValueError("Vectors must have shape (3, 3).")
    a1, a2, a3 = vectors
    V1 = a1.dot(np.cross(a2, a3))
    V2 = a2.dot(np.cross(a3, a1))
    V3 = a3.dot(np.cross(a1, a2))
    assert np.allclose((V1, V2, V3), V1), "Cell volumes V1, V2, V3 not consistent."

    b1 = np.cross(a2, a3) / V1
    b2 = np.cross(a3, a1) / V2
    b3 = np.cross(a1, a2) / V3
    return np.array((b1, b2, b3))


def test_reciprocal_lattice_cubic():
    # simple orthogonal cubic unit cell
    vectors = np.eye(3) * 2.42
    rv = reciprocal_vectors(*vectors)
    rve = explicit_reciprocal_vectors(vectors)
    assert np.allclose(rv, rve)


def test_reciprocal_lattice_hexagonal():
    # hexagonal unit cell taken from Mg cif file
    vectors = np.array([[3.20, 0.0, 0.0], [-1.60, 2.77, 0.0], [0.0, 0.0, 5.12]])
    rv = reciprocal_vectors(*vectors)
    rve = explicit_reciprocal_vectors(vectors)
    assert np.allclose(rv, rve)


def test_reciprocal_lattice_triclinic():
    # triclinic unit cell taken from ReS2 cif file
    vectors = np.array([[6.43, 0.0, 0.0], [-3.38, 5.58, 0.0], [0.20, -1.99, 6.03]])
    rv = reciprocal_vectors(*vectors)
    rve = explicit_reciprocal_vectors(vectors)
    assert np.allclose(rv, rve)


def test_rotate_cell_rotate_reciprocal_vectors():
    ori = Orientation.random()
    # hexagonal unit cell taken from Mg cif file
    vectors = np.array([[3.20, 0.0, 0.0], [-1.60, 2.77, 0.0], [0.0, 0.0, 5.12]])
    vectors_rotated = ori * Vector3d(vectors)
    recip = reciprocal_vectors(*vectors)
    recip_rotated = ori * Vector3d(recip)
    assert np.allclose(reciprocal_vectors(*vectors_rotated.data), recip_rotated.data)


# @pytest.mark.parametrize(params=["CIF_Ni_FCC", "CIF_Ni4W_tetragonal"])
# def test_read_cif_file_and_get_lattice_vectors(file):
#     assert True
# try:
#     a = aseio.read(file)
# except RuntimeError:
#     return  # cif file not loaded/formatted properly
# assert isinstance(a, Atoms)
# va = lattice_vectors_from_structure(a)
# s = loadStructure(str(file))
# assert isinstance(s, Structure)
# vs = lattice_vectors_from_structure(s)
# assert np.allclose(va, vs)
