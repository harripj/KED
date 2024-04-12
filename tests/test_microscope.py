import numpy as np
import pytest

from ked.microscope import electron_wavelength


@pytest.mark.parametrize("voltage, expected", [[200, 2.51], [300, 1.97]])
def test_wavelength(voltage: int, expected: float):
    wavelength = electron_wavelength(voltage)  # voltage in kV, wavelength in Angstrom
    assert np.allclose(round(wavelength * 100, 2), expected)  # expected values in pm
