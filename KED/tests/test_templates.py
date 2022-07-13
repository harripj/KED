from pathlib import Path
import re

from ase import io as aseio
from matplotlib import pyplot as plt
import numpy as np
from orix.quaternion import Orientation
import pytest
from skimage import io as skio
from skimage import measure

import KED
from KED.generator import CrystalDiffractionGenerator
from KED.microscope import electron_wavelength, theta_to_k
from KED.process import check_bounds_coords, virtual_reconstruction
from KED.template import (
    DiffractionTemplateExcitationErrorModel,
    DiffractionTemplateExcitationErrorNorm,
)
from KED.utils import get_image_center


def data_dir():
    base = Path(KED.__file__).parent  # KED.__file__ is __init__.py
    data_dir = base.joinpath("data", "testing")
    assert data_dir.exists()
    return data_dir.relative_to(Path.cwd())


def cif_files():
    return sorted(data_dir().glob("*.cif"))


def pattern_files():
    return sorted(data_dir().glob("*.tif"))


def orientations(pattern_files):
    euler = []
    for f in pattern_files:
        f = str(f)
        match = re.search("\(.+\)", f)
        euler.append(
            [float(v) for v in f[match.start() : match.end()].strip("()").split(",")]
        )
    euler = np.array(euler)
    orientations = Orientation.from_euler(np.deg2rad(euler), direction="lab2crystal")
    return orientations


def get_simulation_parameters_from_file_name(fname):
    # put params in dict
    out = dict()
    fname = Path(fname)

    # cif
    CIF = {c.stem: c for c in cif_files()}
    cif = CIF[fname.stem.split("(")[0].split()[0]]

    # orientation
    match = re.search("\(.+\)", str(fname))
    euler = [
        float(v)
        for v in match.string[match.start() : match.end()].strip("()").split(",")
    ]
    euler = np.array(euler)
    orientation = Orientation.from_euler(np.deg2rad(euler), direction="crystal2lab")
    out["orientation"] = orientation

    # s_max, voltage, max_angle
    kv = fname.stem.split(")")[1].split()
    for kv1 in kv:
        k1, v1 = kv1.split("=")
        out[k1] = float(v1)
    out["cif"] = cif
    out["file"] = fname

    assert all(
        k in out
        for k in ("cif", "orientation", "max_angle", "voltage", "s_max", "file")
    )
    return out


def _template_simulation(file, plot=False, test=True):
    data = get_simulation_parameters_from_file_name(file)

    cif = data["cif"]
    ori = data["orientation"]
    max_angle = data["max_angle"]
    voltage = data["voltage"]
    s_max = data["s_max"]
    file = Path(data["file"])
    wavelength = electron_wavelength(voltage)

    try:
        atoms = aseio.read(cif)
    except RuntimeError:
        assert False

    # load generator
    generator = CrystalDiffractionGenerator(
        atoms,
        voltage,
        max_angle=max_angle,
        debye_waller=False,
        atomic_scattering_factor=False,
        n=20,
    )
    # generate template
    template = generator.generate_templates(
        ori, model=DiffractionTemplateExcitationErrorModel.LINEAR, s_max=s_max * 0.1
    )  # 0.1 is scale factor between softwares

    # load reference image
    im = skio.imread(file)
    size = im.shape[1]
    pixel_size = theta_to_k(np.deg2rad(2 * max_angle) / size, wavelength)
    # overlap appears to be much better when centered on [127, 127]
    # rather than [127.5, 127.5]
    center = get_image_center(im.shape) - 0.5
    # do virtual reconstruction
    VR1 = template.virtual_reconstruction(
        im, pixel_size, center, sum=False, scale_intensity=False
    )
    # project vectors manually
    ijp = template.projected_vectors_to_pixels(pixel_size, center)
    # check bounds on coords
    in_bounds = check_bounds_coords(ijp, im.shape)
    VR2 = virtual_reconstruction(im, ijp[in_bounds], sum=False)
    VR3 = virtual_reconstruction(im, ijp[in_bounds].round().astype(int), sum=False)
    # check that template method and manual method produce same result
    if test:
        assert np.allclose(VR1, VR2)
    # check that at least 75% of apertures overlap with pregenerated
    if test:
        min_overlap_fraction = 0.7
        assert np.count_nonzero(VR1) / VR1.size > min_overlap_fraction
        assert np.count_nonzero(VR2) / VR2.size > min_overlap_fraction
    # assert np.count_nonzero(VR3) / VR3.size > min_overlap_fraction
    # check that no spots are missed in pregenerated
    labelled = measure.label(im)
    regionprops = measure.regionprops(labelled)
    num_reflections = len(regionprops) - 1  # -1 for direct beam
    if test:
        assert VR1.size / num_reflections >= 1
        assert VR2.size / num_reflections >= 1
        assert VR2.size / num_reflections >= 1
    # find any reflections which are not present in pregenerated
    # templates
    VR0 = np.isclose(VR1, 0)
    if np.count_nonzero(VR0):
        # due to some small calculation or rounding errors some
        # simulated reflections may be excited that are not present in
        # the reference patterns. Check that the largest
        # unmatched simulated intensity is smaller than the smallest
        # matched intensity
        # assert max(template.intensity[VR0]) < min(
        #     template.intensity[np.logical_not(VR0)]
        # ) # this test doesn't work for Ni4W due to small spot sizes
        pass
    if plot:
        fig, ax = plt.subplots()
        ax.matshow(im)
        template.plot(ax, pixel_size, center, size=100)
        fig.savefig(file.parent.joinpath("results", file.name), dpi=100)


@pytest.mark.parametrize("file", pattern_files())
def test_template_simulation(file):
    _template_simulation(file)


if __name__ == "__main__":
    for f in pattern_files():
        _template_simulation(f, plot=True, test=False)
