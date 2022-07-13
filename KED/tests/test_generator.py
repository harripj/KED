from pathlib import Path

from ase import io as aseio
from orix.quaternion import Orientation
import pytest
from test_atoms import atoms, cif_Fe_BCC

import KED
from KED.generator import CrystalDiffractionGenerator
from KED.template import DiffractionTemplate, DiffractionTemplateBlock


def test_generator_init(cif_Fe_BCC):
    a = atoms(cif_Fe_BCC)
    generator = CrystalDiffractionGenerator(a, 200)
    assert isinstance(generator, CrystalDiffractionGenerator)
    assert generator.structure == a
    assert generator.voltage * 1e3 == 2e5  # 200 kV
    assert generator.max_angle == 5  # by default


def test_generate_template(cif_Fe_BCC):
    a = atoms(cif_Fe_BCC)
    generator = CrystalDiffractionGenerator(a, 200)
    o = Orientation.random()  # single orientation
    temp = generator.generate_templates(o)
    assert isinstance(temp, DiffractionTemplate)


def test_generate_template_block(cif_Fe_BCC):
    a = aseio.read(cif_Fe_BCC)
    generator = CrystalDiffractionGenerator(a, 200)
    o = Orientation.random((5, 3))
    temp = generator.generate_templates(o)
    assert isinstance(temp, DiffractionTemplateBlock)
    assert temp.shape == o.shape
    assert all(isinstance(i, DiffractionTemplate) for i in temp.ravel())
