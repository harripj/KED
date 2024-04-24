import numpy as np
import pytest
from diffpy.structure import Structure
from orix.quaternion import Orientation, symmetry

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
    o.symmetry = symmetry.D2h
    temp = generator.generate_templates(o)
    assert isinstance(temp, DiffractionTemplate)
    assert np.allclose(temp.orientation.data, o.data)
    assert temp.orientation.symmetry == symmetry.D2h


@pytest.mark.parametrize("grid, shape", [(False, (5, 3)), (True, (5, 5, 5))])
def test_generate_template_block(cif_Fe_BCC, grid, shape):
    generator = CrystalDiffractionGenerator(cif_Fe_BCC, 200)
    if grid:
        o = generate_grid(
            (-1, 1), (-1, 1), (-1, 1), shape[0], as_orientation=True, degrees=True
        )
    else:
        o = Orientation.random(shape)
    o.symmetry = symmetry.Th
    temp = generator.generate_templates(o)
    assert isinstance(temp, DiffractionTemplateBlock)
    assert temp.shape == o.shape
    assert all(isinstance(i, DiffractionTemplate) for i in temp.ravel())
    temp_ori = temp.orientations
    assert isinstance(temp_ori, Orientation)
    assert temp_ori.shape == o.shape
    assert np.allclose(temp_ori.data, o.data)
    assert temp_ori.symmetry == symmetry.Th


@pytest.mark.parametrize("num", [1, 5])
def test_generate_template_block_supersampled(cif_Fe_BCC, num):
    generator = CrystalDiffractionGenerator(cif_Fe_BCC, 200)
    supersampling = 3
    grid = generate_supersampled_grid(
        (-1, 1),
        (-1, 1),
        (-1, 1),
        num=num,
        supersampling=supersampling,
        as_orientation=True,
        degrees=True,
    )
    templates = generator.generate_templates(grid)
    assert isinstance(templates, DiffractionTemplateBlockSuperSampled)
    assert templates.shape == grid.shape
    assert templates.supersampling == (supersampling,) * 3
    for template in templates.ravel():
        assert isinstance(template, DiffractionTemplateBlock)
        assert template.shape == (supersampling,) * 3
