from diffpy.structure import Structure
from orix.quaternion import Orientation

from ked.generator import CrystalDiffractionGenerator, DiffractionGeneratorType
from ked.template import DiffractionTemplate, DiffractionTemplateBlock


def test_generator_init(cif_Fe_BCC):
    generator = CrystalDiffractionGenerator(cif_Fe_BCC, 200)
    assert isinstance(generator, CrystalDiffractionGenerator)
    assert isinstance(generator.structure, Structure)
    assert generator.voltage * 1e3 == 2e5  # 200 kV
    assert generator.max_angle == 5  # by default
    assert generator.kind == DiffractionGeneratorType.CRYSTAL


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
