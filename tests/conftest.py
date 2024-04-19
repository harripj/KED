from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from orix.quaternion import Orientation

from ked.generator import CrystalDiffractionGenerator
from ked.sampling import generate_supersampled_grid
from ked.utils import add_floats_to_array

TEST_DATA_PATH = Path(__file__).parent.joinpath("data")


@pytest.fixture
def size() -> Tuple[int, int]:
    return (128, 128)


@pytest.fixture
def coords() -> NDArray:
    # coords were generated randomly within 128 array without overlap
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
def vals(coords) -> NDArray:
    return np.arange(len(coords)) + 1


@pytest.fixture
def non_overlapping_array(size, coords, vals) -> NDArray:
    arr = np.zeros(size)
    add_floats_to_array(arr, coords, vals)
    return arr


@pytest.fixture
def test_data_path() -> Path:
    return TEST_DATA_PATH


def get_file(file) -> Path:
    path = TEST_DATA_PATH.joinpath(file)
    assert path.exists(), f"{file} not found."
    return path.relative_to(Path.cwd())


@pytest.fixture
def cif_Fe_BCC():
    return get_file("Fe alpha.cif")


@pytest.fixture
def cif_Ni_FCC():
    return get_file("Ni.cif")


@pytest.fixture
def cif_Mg_hexagonal():
    return get_file("Mg.cif")


@pytest.fixture
def cif_ReS2_triclinic():
    return get_file("ReS2.cif")


@pytest.fixture
def cif_Ni4W():
    return get_file("Ni4W.cif")


@pytest.fixture
def cif_files(
    cif_Fe_BCC, cif_Ni_FCC, cif_Mg_hexagonal, cif_ReS2_triclinic, cif_Ni4W
) -> List[Path]:
    return [cif_Fe_BCC, cif_Ni_FCC, cif_Mg_hexagonal, cif_ReS2_triclinic, cif_Ni4W]


@pytest.fixture
def pattern_files(test_data_path: Path):
    return sorted(test_data_path.glob("*.tif"))


@pytest.fixture
def generator(cif_files):
    material = "Fe"
    files = [c for c in cif_files if material in c.stem]
    if not files:
        raise ValueError("No cif files found for material")
    elif len(files) > 1:
        raise ValueError("Multiple cif files found for material")
    file = files[0]
    return CrystalDiffractionGenerator(file, 200)


@pytest.fixture
def template(generator):
    ori = Orientation.random()
    return generator.generate_templates(ori)


@pytest.fixture
def template_block(generator):
    ori = Orientation.random((2, 2))
    return generator.generate_templates(ori)


@pytest.fixture
def template_block_supersampled(generator):
    grid = generate_supersampled_grid(
        (-1, 1),
        (-1, 1),
        (-1, 1),
        num=3,
        supersampling=2,
        as_orientation=True,
        degrees=True,
    )
    return generator.generate_templates(grid)


@pytest.fixture
def diffraction_pattern_shape():
    return (256, 256)


@pytest.fixture
def pixel_size():
    return 0.27  # Angstrom-1
