from diffpy.structure import Structure
import numpy as np
from orix.quaternion import Orientation
import pytest

from ked.generator import CrystalDiffractionGenerator, DiffractionGeneratorType
from ked.sampling import generate_grid, generate_supersampled_grid
from ked.template import (
    DiffractionTemplate,
    DiffractionTemplateBlock,
    DiffractionTemplateBlockSuperSampled,
)


@pytest.mark.parametrize(
    "kV, asf, db",
    [(200, False, False), (300, True, False), (80, False, True), (1_000, True, True)],
)
def test_generator_init(cif_Fe_BCC, kV, asf, db):
    generator = CrystalDiffractionGenerator(
        cif_Fe_BCC, voltage=kV, atomic_scattering_factor=asf, debye_waller=db
    )
    assert isinstance(generator, CrystalDiffractionGenerator)
    assert isinstance(generator.structure, Structure)
    assert generator.voltage == kV
    assert generator.voltage <= 1e3  # voltage is in kV
    assert generator.max_angle == 5  # by default
    assert generator.kind == DiffractionGeneratorType.CRYSTAL
    assert generator.atomic_scattering_factor == asf
    assert generator.debye_waller == db


def test_generate_template(cif_Fe_BCC):
    generator = CrystalDiffractionGenerator(cif_Fe_BCC, 200)
    o = Orientation.random()  # single orientation
    temp = generator.generate_templates(o)
    assert isinstance(temp, DiffractionTemplate)


@pytest.mark.parametrize("grid, shape", [(False, (5, 3)), (True, (5, 5, 5))])
def test_generate_template_block(cif_Fe_BCC, grid, shape):
    generator = CrystalDiffractionGenerator(cif_Fe_BCC, 200)
    if grid:
        grid = generate_grid((-1, 1), (-1, 1), (-1, 1), shape[0])
        o = Orientation.from_axes_angles(
            grid, np.linalg.norm(grid, axis=-1), degrees=True
        )
    else:
        o = Orientation.random(shape)
    temp = generator.generate_templates(o)
    assert isinstance(temp, DiffractionTemplateBlock)
    assert temp.shape == o.shape
    assert all(isinstance(i, DiffractionTemplate) for i in temp.ravel())


def test_generate_template_block_supersampled(cif_Fe_BCC):
    generator = CrystalDiffractionGenerator(cif_Fe_BCC, 200)
    supersampling = 3
    grid = generate_supersampled_grid(
        (-1, 1),
        (-1, 1),
        (-1, 1),
        5,
        supersampling=supersampling,
        as_orientation=True,
        degrees=True,
    )
    templates = generator.generate_template_block(grid)
    assert isinstance(templates, DiffractionTemplateBlockSuperSampled)
    assert templates.shape == grid.shape
    assert templates.supersampling == (supersampling,) * 3
    assert all(isinstance(i, DiffractionTemplateBlock) for i in templates.ravel())
