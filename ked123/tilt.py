from itertools import accumulate
from typing import List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.spatial.transform import Rotation

from .generator import DiffractionGenerator


def compute_VR_tilt_series(
    diffgen: DiffractionGenerator,
    data: List[NDArray],
    rotations: Union[Rotation, List[Rotation]],
    pixel_size: ArrayLike,
    index: int = 0,
    **kwargs,
) -> List[NDArray]:
    """
    Compute virtual reconstructions from tilt series data. The template
    will be rotated for each dataset and the VR will be computed for
    this data.

    Parameters
    ----------
    diffgen: DiffractionGenerator
        The generator which will generate templates.
    data: (N,) list or tuple of ndarray
        A list or tuple of diffraction data.
        The last two axes in each case are the image axes.
    rotations: (N,) list or tuple of rotations
        The rotations that link the datasets in order, ie. links
        data[m-1] to data[m]. The rotation at index should be the
        reference rotation and the others are relative rotations.
    pixel_size: float or (N,) array-like of floats
        Pixel size of data in 1/Angstrom.
        May be a scalar, in which case the same value is used for all
        datasets. If array-like then must be defined for each dataset.
    index: int
        The index of the dataset to which the template applies.
    kwargs:
        Passed to DiffractionGenerator.generate_templates.

    Returns
    -------
    VR: tuple of ndarray
        The computed virtual reconstructions for each dataset.

    """

    assert (
        isinstance(data, (list, tuple)) and len(data) > 1
    ), "data must be a list of more than one dataset."
    assert len(rotations) == len(data), (
        "There must be one rotation between each dataset, currently "
        + f"{len(rotations)} rotations and {len(data)} datasets."
    )
    if isinstance(pixel_size, (list, tuple)):
        assert len(pixel_size) == len(
            data
        ), "pixel_size must either be a scalar or defined for each dataset."
    else:
        pixel_size = (pixel_size,) * len(data)

    if index:
        raise ValueError("index > 0 is currently unsupported.")

    # checked below code manually and works
    rotation_initial = rotations[index]
    rotations_before = rotations[:index]
    rotations_after = rotations[index + 1 :]

    # apply inverse rotations sequentially, ie. from rotation_initial
    # backwards and then reverse list to maintain initial ordering
    rotations_before = list(
        accumulate(
            (rotation_initial, *(r.inv() for r in rotations_before[::-1])),
            lambda x, y: y * x,
        )
    )[1:][::-1]
    # as above but in sequential order and keep initial rotation
    rotations_after = list(
        accumulate((rotation_initial, *rotations_after), lambda x, y: y * x)
    )  # apply rotation sequentially

    rotations_combined = Rotation.from_rotvec(
        np.stack([r.as_rotvec() for r in rotations_before + rotations_after])
    )

    # generate new templates
    templates = diffgen.generate_templates(rotations_combined, **kwargs)

    out = []
    for t, d, p in zip(templates, data, pixel_size):
        # leave center for now, assume blocks are correctly centered
        vr = t.virtual_reconstruction(d, p)
        out.append(vr)

    return out
