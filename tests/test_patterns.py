import numpy as np

from ked.template import (
    DiffractionTemplate,
    DiffractionTemplateBlock,
    DiffractionTemplateBlockSuperSampled,
)
from ked.pattern import DiffractionPattern, DiffractionPatternBlock


def test_pattern(template: DiffractionTemplate, diffraction_pattern_shape, pixel_size):
    pattern = template.generate_diffraction_pattern(
        diffraction_pattern_shape, pixel_size
    )
    assert isinstance(pattern, DiffractionPattern)
    assert pattern.shape == diffraction_pattern_shape
    assert isinstance(pattern.image, np.ndarray)
    assert pattern.image.shape == diffraction_pattern_shape
    assert pattern.orientation


def test_pattern_block(
    template_block: DiffractionTemplateBlock, diffraction_pattern_shape, pixel_size
):
    patterns = template_block.generate_diffraction_patterns(
        diffraction_pattern_shape, pixel_size, progressbar=False
    )
    assert isinstance(patterns, DiffractionPatternBlock)
    assert patterns.shape == template_block.shape
    assert patterns.pattern_shape == diffraction_pattern_shape
    for i, pattern in enumerate(patterns.ravel()):
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape == diffraction_pattern_shape
        if not i:
            assert np.allclose(pattern, patterns[0, 0])
            assert np.shares_memory(pattern, patterns[0, 0])


def test_pattern_block_supersampled(
    template_block_supersampled: DiffractionTemplateBlockSuperSampled,
    diffraction_pattern_shape,
    pixel_size,
):
    patterns = template_block_supersampled.generate_diffraction_patterns(
        diffraction_pattern_shape, pixel_size, progressbar=False
    )
    assert isinstance(patterns, DiffractionPatternBlock)
    assert patterns.shape == template_block_supersampled.shape
    assert patterns.pattern_shape == diffraction_pattern_shape
    for i, pattern in enumerate(patterns.ravel()):
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape == diffraction_pattern_shape
        if not i:
            assert not np.allclose(pattern, patterns[0, 0])
