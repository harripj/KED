from pathlib import Path

import numpy as np
from orix.quaternion.orientation import Orientation
from orix.quaternion.symmetry import C1, Symmetry
import pandas as pd
from scipy.spatial import cKDTree
from skimage import measure, morphology
from tqdm.auto import tqdm

from ..orientations import compute_symmetry_reduced_orientation, convert_to_orix
from .res import ANG

QKEYS = ["a", "b", "c", "d"]


def filter_component_image(mask, min_size=32, exclude_border=0, connectivity=1):
    """
    Use components images to separate and filter components.

    Parameters
    ----------
    mask: ndarray
        Binary component image.
        If mask is 2d then it is assumed to be a 2d image mask.
        If mask is 3d then the first axis is assumed to be a slice axis.
        In this case the maximum is computed along the first axis
        initially.
    min_size: int
        Smallest feature size in pixels to keep.
    exclude_border: int
        Components closer than this value (pixels) to border are
        ignored.
    connectivity: int
        The connectivity order. See skimage documentation.

    Returns
    -------
    mask: ndarray
        The filtered component mask.

    """
    if mask.ndim == 2:
        mask = np.asarray(mask)  # normal 2d mask
    elif mask.ndim == 3:
        mask = np.asarray(mask)  # comes from recreate_components_from_indices
        mask = mask.max(axis=0)

    # create mask and label
    mask = morphology.remove_small_objects(
        mask, min_size=min_size, connectivity=connectivity
    )
    labelled = measure.label(mask)
    rp = measure.regionprops(labelled)
    # filter coords near border
    for r in rp:
        if not (
            all(j >= exclude_border for j in r.centroid)
            and all(
                r.centroid[j] < mask.shape[j] - exclude_border
                for j in range(len(mask.shape))
            )
        ):
            # remove from mask
            mask[r.coords] = 0
    return mask


def create_edge_image(mask, delta=(-1, 1)):
    """
    Creates an image of egdes from a binary or greyscale image eg.
    numbered, skimage.measure.label, or Gwyddion number_grains() image.
    Background must be denoted with zeroes.

    The original image is rolled by delta typically ±1 to find the edges
    in all of the images dimensions. The labelling of the original image
    is preserved.

    Parameters
    ----------
    image: numpy.ndarray
        Often a 2d (but can be 1d or 3d) array. Typically this image is
        greyscale, ie. skimage.measure.label.
    delta: tuple of ints
        The shift over which to find the edges, typically ±1.
        Default is (-1,1).

    Returns
    -------
    edges: array
        The found edges, maintains the original labelling.
        Same shape as input array.

    """
    # create empty array to hold found edges
    edges = np.zeros_like(mask)

    for axis in np.arange(mask.ndim):  # works for 2d and 3d
        for shift in delta:
            # roll the image along one direction by shift in delta
            shifted_image = np.roll(mask, shift, axis=axis)
            # the edges of the grains are found where the data is non-zero
            # in the original image and zero after rolling by ± 1
            coords = np.nonzero(np.logical_and(mask, shifted_image != mask))
            # this preserves the original grain numbering system
            edges[coords] = mask[coords]

    return edges


def compute_component_walker(ori, start, angle=3.0, symmetry=None):
    """
    Compute connected orientations into components.
    A walker is initialized to search for small misorientations between
    connected pixels.

    Parameters
    ----------
    ori: orix.quaternion.Orientation
        The orientation map.
    start: tuple of ints
        The starting coordinate for the walker in ori.
    angle: float
        The maximum angle in dgerees between neighbouring pixels to
        establish a connection.
    symmetry: orix.quaternion.Symmetry
        The symmetry of the map.

    Returns
    -------
    component: ndarray
        The connected component mask
    """
    ori = convert_to_orix(ori)
    if not isinstance(symmetry, Symmetry):
        raise ValueError("symmetry must be orix.quaternion.Symmetry.")

    # mask keeps track of the walker
    mask = np.zeros(ori.shape, dtype=int)
    mask[start] = 1
    mask_temp = np.copy(mask)

    # the dilation kernel disk(1) is a cross
    selem = morphology.disk(1)

    while np.count_nonzero(mask) < mask.size:
        # get the edge coordinates of the walker shape
        edge_coords = np.column_stack(np.nonzero(create_edge_image(mask)))
        # only want to compare misori to closest valid coords
        edge_coords = edge_coords[mask[tuple(edge_coords.T)] == 1]
        tree = cKDTree(edge_coords)
        # expand search area
        morphology.binary_dilation(mask, selem=selem, out=mask_temp)
        # get new coords
        coords = np.nonzero(np.logical_xor(mask, mask_temp))
        # get closest pixel pairing
        dist, index = tree.query(np.column_stack(coords))
        # apply filtering here
        # 0 = not visited, 1 = visited okay, 2 = visited not okay
        misori = (
            ori[coords].outer(symmetry)
            * ~ori[tuple(edge_coords[index].T)][:, np.newaxis]
        ).angle.data.min(axis=1)
        vals = misori <= np.deg2rad(angle)
        # update search grid with new positions
        mask[coords] = (~vals).astype(mask.dtype) + 1

    return mask == 1  # return only connected mask


def _convert_ang_input(ang, check_shape=True):
    """Check input are all ANG or Path to .ang and return list of ANG."""
    if not isinstance(ang, (list, tuple)):
        assert isinstance(ang, (ANG, str, Path)), "ang must be ANG or path to .ang."
        ang = [ang]

    if all(isinstance(a, (str, Path)) for a in ang):
        ang = [ANG.from_file(a) for a in ang]

    if not all(isinstance(a, ANG) for a in ang):
        raise ValueError("ang are not all ANG.")

    if check_shape:
        if not len(set(a.shape for a in ang)) == 1:
            raise ValueError("ANG do not have the same shape.")

    return ang


def orientation_stack_from_ang(ang):
    """
    Compute orientation stack from ANG.

    Parameters
    ----------
    ang : (N,) iterable of ANG or path to .ang
        ANG to stack orientations from.
        All ANG must have same shape (L, M)

    Returns
    -------
    ori: (N, L, M) orix.quaternion.Orientation
        Stacked orientations.
    """
    # check consistent ANG shapes
    ang = _convert_ang_input(ang, check_shape=True)
    return Orientation(np.stack([a.orientations.data for a in ang]))


def connect_components(
    ang,
    max_angle=5.0,
    symmetry=C1,
    min_size=32,
    exclude_border=0,
    connectivity=1,
    save=None,
):
    """
    Connect a set of indexed orientations (.ang) within a defined
    angular tolerance.

    Parameters
    ----------
    ang: (N,) iterable of (M, N, P) orix.quaternion.Orientation
        Orientations to connect components from.
    max_angle: float
        The maximum misorientation angle used to connect components in
        degrees.
    symmetry: orix.quaterion.Symmetry
        The crystal symmetry.
    min_size: int
        Smallest feature size in pixels to keep.
    exclude_border: int
        Components closer than this value (pixels) to border are
        ignored.
    connectivity: int
        The connectivity order. See skimage documentation.
    save: None or str or Path
        Component indices will be saved to this file if provided.
        File type will be .txt.

    Returns
    -------
    components: list of ndarray
        List of indices to reconstruct the components.

    """
    # convert inputs and get ANG shape
    ang = _convert_ang_input(ang, check_shape=True)
    shape = ang[0].shape

    # only compute valid points
    mask1 = np.stack([a.mask for a in ang])
    indices1 = np.column_stack(np.nonzero(mask1))
    # angles = np.stack([a.angles.reshape(shape + (3,)) for a in ang])[mask1]
    reliability = np.stack([a.reliability for a in ang])[mask1]

    # compute all symmetry points for all valid points once
    if not isinstance(symmetry, Symmetry):
        raise TypeError("symmetry must be orix.quaternion.Symmetry.")
    ori = Orientation(np.stack([a.orientations.data for a in ang])[mask1])
    ori.symmetry = symmetry

    # generate mask as progress counter which is updated each loop
    mask = np.ones(np.count_nonzero(mask1), dtype=bool)
    # get maximum disorientation in radians
    max_angle_rad = np.deg2rad(max_angle)

    out = []
    # loop until all pixels are accounted for
    progressbar = tqdm(total=mask.size)
    while np.count_nonzero(mask):
        indices = np.nonzero(mask)[0]
        # get most reliable point
        index = reliability[mask].argmax()
        # ...and its orientation
        ori_curr = ori[mask][index]
        assert ori_curr.size == 1
        # calculate misorientation between this orientation and rest of
        # the map
        angles = ori[mask].angle_with(ori_curr)
        # find points closer than max_angle
        mask_temp = angles <= max_angle_rad
        # update mask for next iteration
        # mask is False where points have already been selected
        mask[indices] = np.logical_not(mask_temp)
        component_indices = indices1[indices[mask_temp]]

        # now filter component images from noise using 2d size filter
        component_mask = recreate_component_from_indices(
            component_indices, shape, len(ang)
        )
        filtered_component_mask = filter_component_image(
            component_mask,
            min_size=min_size,
            exclude_border=exclude_border,
            connectivity=connectivity,
        )
        # separate non-connected components that have similar orientation
        rp = measure.regionprops(
            measure.label(filtered_component_mask, connectivity=connectivity)
        )
        # get indices for each separated components within component image
        for r in rp:
            coords3d = np.concatenate(
                [
                    np.column_stack([np.full(r.area, j), r.coords])
                    for j in range(len(ang))
                ]
            )
            out.append(coords3d[component_mask[tuple(coords3d.T)]])

        # increase progressbar with number of new pixels tested
        progressbar.update(np.count_nonzero(mask_temp))

    # sort by largest grain first
    out = sorted(out, key=len)[::-1]

    if isinstance(save, (str, Path)):
        save = Path(save)
        with open(save, "w") as f:
            # add ANG file names to file
            for a in ang:
                f.write(f"### {str(a.file)}\n")
            for i, indices in enumerate(out):
                f.write(f"# {i}\n")
                np.savetxt(f, indices, fmt="%i")
        print(f"Components indices saved: {str(save)}.")

    return out


def load_components_indices(fname):
    """Load components indices from file. Returns these indices and ANG
    file names."""
    fname = Path(fname)
    with open(fname, "r") as f:
        out = []
        ang_files = []
        # None for first set of indices
        temp = None
        for i, line in enumerate(f):
            if line.startswith("###"):
                ang_files.append(line.replace("#", "").strip())
            elif line.startswith("# "):
                if temp is not None:
                    out.append(np.array(temp))
                temp = []
            else:
                temp.append([int(j) for j in line.strip().split()])
        # last read set of indices need to be added to out
        out.append(np.array(temp))
    return out, ang_files


def load_components(fname):
    """Load components file created by calculate_component_statistics."""
    return pd.read_csv(fname, index_col=0)


def load_components_ASTAR(fname):
    """Load component file produced from ASTAR MapViewer."""
    with open(fname) as f:
        column_names = f.readline()
    column_names = [n for n in column_names.split() if "," not in n]
    return pd.read_csv(fname, delimiter="\s+", names=column_names, skiprows=1)


def recreate_component_from_indices(indices, shape, n=None):
    """
    Recreate a (binary) component or grain from a set of indices
    corresponding to n overlapping datasets.

    Parameters
    ----------
    indices: (N, 3) ndarray
        The indices of the points representing the image.
        The first column represents the .ang file index.
        The last two columns represent the row and column index of the
        grain in the image.
    shape: tuple
        The shape of resulting image (original dataset shape).
    n: int or None
        The number of slices (.ang or indexing passes) used to calculate
        the component. If None then max(indices[:, 0]) + 1 (slice
        dimension) is used.

    Returns
    -------
    image: (M, shape) ndarray
        The recreated grain image.
        The first (M) dimension is the slice dimension (for
        multi-indexing).
    """

    indices = np.asarray(indices)
    assert indices.shape[1] == 3, "Indices must be (N, 3) ndarray."

    if n is None:
        n = indices[:, 0].max() + 1

    out = np.zeros((n,) + shape, dtype=bool)
    out[tuple(indices.T)] = 1
    return out.squeeze()


def calculate_component_statistics(ori, indices, save=None, round=2):
    """
    Calculate component statistics from component indices.

    Parameters
    ----------
    ori: (L, P, Q) orix.quaternion.Orientation
        Orientation data needed to determine average orientation.
        Maybe be loaded from .ang using `orientation_stack_from_ang`.
        Symmetry property should be defined.
    indices: (N, 3) ndarray or (M,) iterable of (N, 3) ndarray
        The indices used to create the component from the ANG data.
        Columns are (slice, i, j).
    symmetry: None or orix.quaternion.Symmetry
        The symmetry of the orientations. Default is C1 (identity).
    save: None or str or Path
        If provided the component statistics will be save to this file
        in CSV format. The file type will be .csv.
        The data included in the file for each component:
        - a, b, c, d: Mean orientation quaternion values
          (orix.quaternion.Quaternion)
        - psi1, phi, psi2: Euler angles in degrees (Bunge convention)
          in 'lab2crystal' reference frame (orix.quaternion.Orientation)
        - area: The component size in number of pixels
        - i, j: Component position in row (i) and column (j) format
        - x, y: Component position in Cartesian format
    round: int
        Saved data will be rounded to this many decimal places.
        Quaternion data will not be rounded.

    Returns
    -------
    ori_mean: (M,) orix.quaternion.Orientation
        Component mean orientation.
    ij: (M, 2) ndarray
        Component mean projected position.
    size: (M,) ndarray
        The component projected area in pixels.
    """

    if not isinstance(ori, Orientation):
        raise TypeError("ori must be orix.quaternion.Orientation.")
    if ori.ndim != 3:
        raise ValueError("ori must be 3 dimensional.")

    err_indices = "indices must (N, 3) ndarray or iterable of (N, 3) ndarray."
    if isinstance(indices, np.ndarray):
        if indices.shape[-1] == 3:
            # in correct format
            indices = (indices,)
        else:
            raise ValueError(err_indices)

    if isinstance(indices, (list, tuple)):
        # compute many statistics
        ori_mean = []
        ij_mean = []
        area = []
        for idx in indices:
            idx = np.asarray(idx)
            assert idx.ndim == 2 and idx.shape[1] == 3, err_indices
            # get the orientations
            o1 = ori[tuple(idx.T)]
            # choose orientation closest to center as a reference
            o2 = o1[np.linalg.norm(idx - idx.mean(axis=0), axis=-1).argmin()]
            # get symmetry of o1 with smallest angle to o2
            o1 = compute_symmetry_reduced_orientation(o1, o2)
            ori_mean.append(o1.mean())
            # only care about unique projected coordinates
            # avoids averaging over possible duplicate coordinates on
            # multiple layers
            unique = np.unique(idx[:, 1:], axis=0)
            ij_mean.append(unique.mean(axis=0))
            area.append(len(unique))
    else:
        raise TypeError(err_indices)

    # get mean orientation and mean projected ij position
    ori_mean = Orientation([o.data for o in ori_mean]).squeeze()
    ij_mean = np.squeeze(ij_mean)
    area = np.squeeze(area)

    if isinstance(save, (str, Path)):
        save = Path(save)
        euler = np.rad2deg((ori_mean).to_euler())
        test = Orientation.from_euler(np.deg2rad(euler), direction="lab2crystal")
        dist = (test - ori_mean).angle.data
        if not np.allclose(dist, 0):
            print("Not all Euler angles may be consistent with quaternions.")

        euler = euler.round(round)
        ij_mean_rounded = ij_mean.round(round)

        d = dict(
            a=ori_mean.a,
            b=ori_mean.b,
            c=ori_mean.c,
            d=ori_mean.d,
            psi1=euler[:, 0],
            phi=euler[:, 1],
            psi2=euler[:, 2],
            area=area,
            i=ij_mean_rounded[:, 0],
            j=ij_mean_rounded[:, 1],
            x=ij_mean_rounded[:, 1],
            y=ij_mean_rounded[:, 0],
        )
        df = pd.DataFrame(data=d)
        df.to_csv(save.with_suffix(".csv"))
        print(f"Components file saved: {save}")

    # figure out what to return
    return ori_mean, ij_mean, area
