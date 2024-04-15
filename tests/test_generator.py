from diffpy.structure import Structure
from orix.quaternion import Orientation
import pytest

from ked.generator import CrystalDiffractionGenerator, DiffractionGeneratorType
from ked.template import DiffractionTemplate, DiffractionTemplateBlock


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


def test_generate_template_block(cif_Fe_BCC):
    generator = CrystalDiffractionGenerator(cif_Fe_BCC, 200)
    o = Orientation.random((5, 3))
    temp = generator.generate_templates(o)
    assert isinstance(temp, DiffractionTemplateBlock)
    assert temp.shape == o.shape
    assert all(isinstance(i, DiffractionTemplate) for i in temp.ravel())
