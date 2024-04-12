from pathlib import Path
import re
from typing import List, Union

from ked123.generator import CrystalDiffractionGenerator
from ked123.microscope import electron_wavelength, theta_to_k
from ked123.process import check_bounds_coords, virtual_reconstruction
from ked123.template import DiffractionTemplateExcitationErrorModel
from matplotlib import pyplot as plt
import numpy as np
from orix.quaternion import Orientation
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage import io as skio
from skimage import measure, morphology, segmentation


@pytest.fixture
def orientations(pattern_files):
    euler = []
    for f in pattern_files:
        f = str(f)
        match = re.search(r"\(.+\)", f)
        euler.append(
            [float(v) for v in f[match.start() : match.end()].strip("()").split(",")]
        )
    euler = np.array(euler)
    orientations = Orientation.from_euler(np.deg2rad(euler), direction="lab2crystal")
    return orientations


def get_simulation_parameters_from_file_name(
    fname: Union[str, Path], cif_files: List[Path]
):
    # put params in dict
    out = dict()
    fname = Path(fname)

    # cif
    CIF = {c.stem: c for c in cif_files}
    cif = CIF[fname.stem.split("(")[0].split()[0]]

    # orientation
    match = re.search(r"\(.+\)", str(fname))
    euler = [
        float(v)
        for v in match.string[match.start() : match.end()].strip("()").split(",")
    ]
    euler = np.array(euler)
    orientation = Orientation.from_euler(np.deg2rad(euler))
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


def _template_simulation(
    data: dict, plot: bool = False, test: bool = True, min_overlap: float = 0.75
):
    cif = data["cif"]
    ori = data["orientation"]
    max_angle = data["max_angle"]
    voltage = data["voltage"]
    s_max = data["s_max"]
    file = Path(data["file"])
    wavelength = electron_wavelength(voltage)

    # load generator
    generator = CrystalDiffractionGenerator(
        cif,
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
    if cif.stem.startswith("Ni4W") or cif.stem.startswith("ReS2"):
        # Ni4W has a very small spot size
        im = morphology.dilation(im, footprint=morphology.disk(2))
    size = im.shape[1]
    pixel_size = theta_to_k(np.deg2rad(2 * max_angle) / size, wavelength)
    # overlap appears to be much better when centered on [127, 127]
    # rather than [127.5, 127.5]
    center = tuple(s / 2 for s in im.shape)
    # do virtual reconstruction
    VR1 = template.virtual_reconstruction(
        im, pixel_size, center, sum=False, scale_intensity=False
    )
    # project vectors manually
    ijp = template.projected_vectors_to_pixels(pixel_size, center)
    in_bounds = check_bounds_coords(ijp, im.shape)
    ijp_in_bounds = ijp[in_bounds]
    VR2 = virtual_reconstruction(im, ijp_in_bounds, sum=False)
    # check that template method and manual method produce same result
    if test:
        assert VR1.size == VR2.size
        assert np.allclose(VR1, VR2, atol=1e-6)
    # check minimum aperture overlap with pregenerated
    if test:
        assert (np.count_nonzero(VR1) / VR1.size) > min_overlap
        assert (np.count_nonzero(VR2) / VR2.size) > min_overlap
    # find any reflections which are not present in pregenerated templates
    mask_VR2_no_intensity = np.isclose(VR2, 0)
    if np.count_nonzero(mask_VR2_no_intensity):
        # due to some small calculation or rounding errors some
        # simulated reflections may be excited that are not present in
        # the reference patterns. Check that the largest
        # unmatched simulated intensity is smaller than the smallest
        # matched intensity
        # NB: this test doesn't work for Ni4W due to small spot sizes
        assert np.max(VR2[mask_VR2_no_intensity]) < np.min(VR2[~mask_VR2_no_intensity])
    # remove border reflections which cause errors for this test
    im_cb = segmentation.clear_border(im, buffer_size=2)
    # check that no spots are missed in pregenerated
    labelled, num_reflections = measure.label(im_cb, return_num=True)
    num_reflections -= 1  # -1 for direct beam
    if test:
        assert VR1.size / num_reflections >= 1
        assert VR2.size / num_reflections >= 1
    if test and voltage < 300:
        # check distance between simulated and pregenerated reflections
        rp = measure.regionprops(labelled)
        # some reflections with same g vector are connected in the image, ignore these
        labelled_ij = [r.centroid for r in rp if r.eccentricity < 0.5]
        # remove direct beam
        direct_index = np.linalg.norm(np.array(labelled_ij) - center, axis=-1).argmin()
        labelled_ij.pop(direct_index)
        dist = cdist(ijp_in_bounds, labelled_ij)
        r, c = linear_sum_assignment(dist)
        tol = 4
        assert np.all(dist[r, c] <= tol)

    if plot:
        fig, ax = plt.subplots()
        ax.matshow(im)
        template.plot(ax, pixel_size, center, size=100)
        _dir = file.parent.joinpath("results")
        if not _dir.exists():
            _dir.mkdir()
        fig.savefig(_dir.joinpath(file.name), dpi=100)


def test_template_simulation(pattern_files, cif_files):
    count = 0
    for i, file in enumerate(pattern_files):
        if file.stem.startswith("ReS2"):
            continue
        min_overlap = 0.8 if file.stem.startswith("Ni4W") else 0.85
        data = get_simulation_parameters_from_file_name(file, cif_files)
        try:
            _template_simulation(data, plot=False, test=True, min_overlap=min_overlap)
        except Exception as e:
            assert not file
        count += 1
    assert i
    assert count
