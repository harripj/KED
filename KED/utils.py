from itertools import product
from pathlib import Path
from typing import Tuple, Union

from matplotlib import pyplot as plt
from ncempy.io import mrc
import numba
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from orix.quaternion import symmetry as osymmetry
from orix.vector import AxAngle, Vector3d
import pandas as pd
from scipy import constants, ndimage
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

# define dtype for whole module regarding simulations etc.
DTYPE = np.float32


def calculate_zone_axis(ori, n=15):
    """Return approximate Z zone axis and misorientation (in radians)."""
    # find the vector which maps to z after rotation
    z = (~ori * Vector3d.zvector()).data
    # the smallest factor should be a multiple of the smallest component
    z_min = np.abs(z).min()
    z_arr = z / z_min

    # search for best integer axis
    mult = np.outer(np.arange(1, n), z_arr)
    mult_rounded = np.round(mult)
    diff = mult - mult_rounded
    err = np.linalg.norm(diff, axis=-1)

    # find best rounded axis
    axis = Vector3d(mult_rounded[err.argmin()])
    # get the misorientation between rounded axis and z
    dp = np.dot((ori * axis).unit.data.ravel(), Vector3d.zvector().unit.data.ravel())
    misori = np.arccos(dp)

    return axis, misori


def generate_thetas(num, min=0, max=2.0 * np.pi):
    """Convenience function to generate theta array with a defined
    number of points.
    """
    return np.linspace(min, max, num, endpoint=False)


def get_image_center(shape, center=None):
    """Get image center from a given shape.

    Parameters
    ----------
    shape: 2-tuple
        The image shape.
    center : None or 2-tuple
        If defined then center is simply returned.
        If None then the center of the image is calculated.

    Returns
    -------
    center: (2,) ndarray
        The center of the image.
    """
    if len(shape) != 2:
        raise ValueError("Shape must have length 2.")
    if center is None:
        center = (np.asarray(shape) - 1) / 2
    if len(center) != 2:
        raise ValueError("Center must have length 2.")
    return np.asarray(center)


def fibonacci_sphere_angular_resolution_to_n(resolution):
    """
    Calculate approximate number of points needed to generate a
    fibonacci sphere with the desired angular resolution.

    Parameters
    ----------
    resolution: float
        Angular resolution in degrees.

    Returns
    -------
    n: int
        Approximate number of points.
    """
    # calculated median angular res. in deg.
    res = np.array([6.106, 1.935, 0.588, 0.107, 0.013])
    exp = np.arange(3, 8)  # exponential factor

    # interpolate trend and extract number of pts
    f = interp1d(res, exp)
    n = f(resolution)

    return int(10**n)


def fibonacci_sphere(n=None, res=None, offset=0.5):
    """
    Produce n points approximately evenly spaced on a sphere.

    Parameters
    ----------
    n: int or None
        Number of points.
    res: float or None
        Desired angular resolution.
    offset: float
        Offset for fibonacci spiral.
        This value will generate different sequences.

    Returns
    -------
    xy: (N, 3) ndarray
        (x, y, z) coordinates.

    Notes
    -----
    [1] https://stackoverflow.com/a/44164075/12063126

    """
    if n is None and res is None:
        raise ValueError("Either n or res must not be None.")
    elif n is not None and res is not None:
        raise ValueError("Only one of n or res must not be None.")
    else:
        if res is not None:
            n = fibonacci_sphere_angular_resolution_to_n(res)
        n = int(n)

    i = np.arange(n, dtype=np.float32) + offset

    phi = np.arccos(1 - 2 * i / n)
    theta = 2 * np.pi * constants.golden * i

    return np.column_stack(
        (np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi))
    )


def get_orientations_standard_triangle(res=1, v1=(0, 0, 1), v2=(1, 0, 1), v3=(1, 1, 1)):
    """
    Return approximately evenly spaced orientations within the standard
    triangle.

    Parameters
    ----------
    res: float
        The approximate angular point resolution in degrees.
    v1, v2, v3: array-like
        Miller index vertices defining the triangle.

    Returns
    -------
    ori: Rotation
        Orientations within the standard triangle.

    """
    if res == 1:
        n = 40000
    else:
        raise ValueError("Currently only res=1 is implemented.")

    pts = np.column_stack(fibonacci_sphere(n))

    # vertices defining the standard triangle
    verts = np.array((v1, v2, v3))

    # use center point to determine polarity of dot product
    center = verts.mean(axis=0)
    center = center / np.linalg.norm(center)

    for i, v in enumerate(verts):
        _cross = np.cross(verts[(i + 1) % len(verts)], v)
        temp = np.dot(pts, _cross)

        _cross_val = np.dot(center, _cross)
        if _cross_val > 0:
            pm = 1
        elif _cross_val < 0:
            pm = -1
        else:
            raise ValueError(f"Cannot work out direction.")

        # filter points
        pts = pts[pm * temp >= 0]

    # find the rotation vector that operates on (001)
    rot = np.cross((0, 0, 1), pts)
    angle = np.arcsin(np.linalg.norm(rot, axis=-1))

    # normalise rot product vector and multiply by angle in rad. -> rotvec
    rot = (rot.T / np.linalg.norm(rot, axis=-1)).T
    return Rotation.from_rotvec((rot.T * angle).T)


def plot_solution_grid(ax, soln, skew, grid_range, n, cmap=plt.cm.plasma):
    """
    Plot 3D orientation solution grid on axes.

    Parameters
    ----------
    ax: plt.Axes
    soln: ndarray
        3D solution.
    skew: ndarray
        The skew of the ground truth in degrees.
    grid_range: float
        The angular range of the soltion grid in degrees (normally ±).
    n: int
        The grid shape.
    cmap: plt.ColorMap.

    """
    assert soln.ndim == 3 and soln.shape == (n, n, n)
    ax.voxels(soln, facecolors=cmap(soln / soln.sum()))

    # show skew on grid
    _skew = ((skew + grid_range) / (2 * grid_range)) * n  # 0.5 to center in voxel
    ax.scatter(
        *_skew + 0.5, color="red", label=f"Skew: {tuple(round(j, 3) for j in skew)}"
    )
    ax.scatter(*_skew + 0.5, edgecolor="red", fc="None", s=1000)

    ax.set_xticks([0, n / 2, n])
    ax.set_yticks([0, n / 2, n])
    ax.set_zticks([0, n / 2, n])

    ax.set_xticklabels([-grid_range, 0, grid_range])
    ax.set_yticklabels([-grid_range, 0, grid_range])
    ax.set_zticklabels([-grid_range, 0, grid_range])

    # calculate CoM
    CoM = np.array(ndimage.center_of_mass(soln))
    ax.scatter(
        *CoM + 0.5,
        color="b",
        label=f"CoM: {tuple(round(j, 3) for j in CoM / n * 2 * grid_range - grid_range)}",
    )
    ax.scatter(*CoM + 0.5, edgecolor="blue", fc="None", s=1000)

    ax.legend(loc="lower right")


def create_VR_crystal_map(files, kind="max"):
    """
    Use many virtually reconstructed files to create an overall crystal
    map.

    Parameters
    ----------
    files: str or Path
        The .mrc files to open.
    kind: str
        Image processing calculation type.
        Either 'max' or 'sum'.

    Returns
    -------
    image: ndarray
        The VR crystal map for each dataset (axis 0 in image).
    """

    kind = kind.lower()
    if kind not in ("sum", "max"):
        raise ValueError("kind must either be 'sum' or 'max'.")

    for i, f in enumerate(files):
        # open files
        _mrc = mrc.mrcReader(f)
        _data = _mrc["data"]
        # scale data by max per frame
        _data = (_data.T / _data.max(axis=(-2, -1))).T
        # accumulate data
        if not i:
            data = _data
        else:
            temp = np.stack((data, _data), axis=-1)
            if kind == "sum":
                data = temp.sum(axis=-1)
            elif kind == "max":
                data = temp.max(axis=-1)
            else:
                assert False, "This should have already been handled."

    return data


def read_links(fname):
    """
    Read a link file generated by 3D-SPED.

    Parameters
    ----------
    fname: str or Path
        Path to .link file.

    Returns
    -------
    data:
    header: dict
        Generation parameters.
    """
    header = dict()

    current_header = None
    datasets = []

    with open(Path(fname), "r") as f:
        for i, line in enumerate(f):
            if line.strip().endswith(":"):
                current_header = line.strip()[:-1]
            else:
                if current_header.upper() == "DATASETS":
                    datasets.append(line.strip())
                elif current_header.upper() == "PARAMETERS":
                    key, val = line.strip().split("\t")
                    # only expect numeric values here currently
                    try:
                        val = int(val)
                    except ValueError:
                        val = float(val)
                    header[key] = val
                elif current_header.upper() == "DATA":
                    break
    header["DATASETS"] = datasets

    return pd.read_csv(Path(fname), skiprows=i, delimiter="\t"), header


def get_series_name(fname):
    """
    Attempt to retrieve base series name from an ASTAR file.
    Various filename extensions will be removed.

    Parameters
    ----------
    fname: str or Path
        Path to ASTAR generated file.

    Returns
    -------
    series: str
        The base series name.

    """
    fname = Path(fname)
    base = fname.stem

    # try to remove extensions added from ASTAR
    if "_Edited" in base:
        base = base[: base.index("_Edited")]

    if "(full)" in base:
        base = base[: base.index("(full)") - 1]

    if base.endswith("_C"):
        base = base[:-2]

    if base.endswith("_CROP"):
        base = base[:-5]

    return base.strip()


def get_angle_from_series_name(fname, delimiter="_"):
    """Attempt to get angle from series name.

    This is based on a TVIPS file naming system,
    eg. rec_0 deg_20210722_125759_000 (full)_C.blo.
    """
    fname = get_series_name(fname)
    angle_str = [s.lower() for s in fname.split(delimiter)]

    for i, s in enumerate(angle_str):
        if "deg" in s:
            break
    else:
        i = -1
    angle_str = angle_str[i]

    if "deg" in angle_str:
        angle_str = angle_str.replace("deg", "")
    if "," in angle_str:
        angle_str = angle_str.replace(",", ".")

    if "m" in angle_str:
        angle_str = angle_str.replace("m", "")
        factor = -1
    elif "p" in angle_str:
        angle_str = angle_str.replace("p", "")
        factor = 1
    else:
        factor = 1

    try:
        val = float(angle_str) * factor
    except ValueError:
        # not numeric
        val = None
    return val


def load_state(fname, parse_values=True):
    """
    Load a .tltstate file.

    Parameters
    ----------
    fname : Path or str
        .tltstate filename.
    parse_values : bool
        If True then values are fomatted, eg. to ndarray, before being
        returned.

    Returns
    -------
    info : dict
        The state information.
    """
    base_directory = None
    datasets = []
    results = []
    components = []
    angles = []
    rotvecs = []
    symmetry = None
    pixel_size = []
    cif = None
    diffraction_generator_parameters = []
    scan_transform_file = None
    crystal_map = []
    notes = []

    current_header = None
    strip_chars = " :\n"

    with open(fname, "r") as f:
        for line in f:
            if line.upper().startswith("BASE DIRECTORY:"):
                current_header = line.upper().strip(strip_chars)
            elif line.upper().startswith("DATASETS:"):
                current_header = line.upper().strip(strip_chars)
            elif line.upper().startswith("RESULTS:"):
                current_header = line.upper().strip(strip_chars)
            elif line.upper().startswith("COMPONENTS:"):
                current_header = line.upper().strip(strip_chars)
            elif line.upper().startswith("ANGLES:"):
                current_header = line.upper().strip(strip_chars)
            elif line.upper().startswith("SYMMETRY:"):
                current_header = line.upper().strip(strip_chars)
            elif line.upper().startswith("ROTVECS:"):
                current_header = line.upper().strip(strip_chars)
            elif line.upper().startswith("CIF:"):
                current_header = line.upper().strip(strip_chars)
            elif line.upper().startswith("DIFFRACTION GENERATOR PARAMETERS:"):
                current_header = line.upper().strip(strip_chars)
            elif line.upper().startswith("PIXEL SIZE:"):
                current_header = line.upper().strip(strip_chars)
            elif line.upper().startswith("SCAN TRANSFORM FILE:"):
                current_header = line.upper().strip(strip_chars)
            elif line.upper().startswith("CRYSTAL MAP:"):
                current_header = line.upper().strip(strip_chars)
            elif line.upper().startswith("NOTES:"):
                current_header = line.upper().strip(strip_chars)

            else:
                # only strip newline chars below
                if current_header == "BASE DIRECTORY":
                    base_directory = line.strip()
                elif current_header == "DATASETS":
                    datasets.append(line.strip())
                elif current_header == "RESULTS":
                    results.append(line.strip())
                elif current_header == "COMPONENTS":
                    components.append(line.strip())
                elif current_header == "ANGLES":
                    angles.append(line.strip())
                elif current_header == "SYMMETRY":
                    symmetry = line.strip()
                elif current_header == "ROTVECS":
                    rotvecs.append(line.strip())
                elif current_header == "CIF":
                    cif = line.strip()
                elif current_header == "DIFFRACTION GENERATOR PARAMETERS":
                    diffraction_generator_parameters.append(line.strip())
                elif current_header == "PIXEL SIZE":
                    pixel_size.append(line.strip())
                elif current_header == "SCAN TRANSFORM FILE":
                    scan_transform_file = line.strip()
                elif current_header == "CRYSTAL MAP":
                    crystal_map.append(line.strip())
                elif current_header == "NOTES":
                    notes.append(line.strip())
                else:
                    # this should not happen
                    assert (
                        current_header is not None
                    ), "Current header is None, badly formatted file."
                    raise ValueError(f"Unknown error caused by line: {line}.")

    if parse_values:
        angles = np.array([float(a) for a in angles])
        symmetry = getattr(osymmetry, symmetry)
        rotvecs = AxAngle(
            [tuple(float(v) for v in r.strip("()").split(",")) for r in rotvecs]
        )
        pixel_size = np.array([float(p) for p in pixel_size])
    out = dict(
        BASE_DIRECTORY=base_directory,
        DATASETS=datasets,
        RESULTS=results,
        COMPONENTS=components,
        ANGLES=angles,
        ROTVECS=rotvecs,
        SYMMETRY=symmetry,
        PIXEL_SIZE=pixel_size,
        CIF=cif,
        DIFFRACTION_GENERATOR_PARAMETERS=diffraction_generator_parameters,
        SCAN_TRANSFORM_FILE=scan_transform_file,
        CRYSTAL_MAP=crystal_map,
        NOTES=notes,
    )

    return out


def get_angle_from_filename(fname: Union[str, Path], delimiter: str = " "):
    """
    Work out tilt angle from file name.
    Handles typical suffixes from data acquisition 'm, p, C' etc.

    Parameters
    ----------
    fname: str of Path
        File name.
    delimiter: str
        The file basename delimiter, typically space or underscore.

    Returns
    -------
    angle: float or None
        The read angle or None if not properly found.

    """
    fname = Path(fname)
    stem = fname.stem.split(delimiter)

    # typically suffixed
    part = stem[-1].upper()

    if part == "_C" or part == "C":
        # in the case of centered blockfiles
        part = stem[-2].upper()  # should be next part

    try:
        if part.endswith("M") or part.startswith("M"):
            norm = -1
            angle = float(part.replace("M", ""))
        elif part.endswith("P") or part.startswith("P"):
            norm = 1
            angle = float(part.replace("P", ""))
        else:
            norm = 1
            angle = float(part)

        out = angle * norm
    except ValueError:
        out = None

    return out


@numba.njit
def _unravel_index_2d(index, shape):
    """Unravel flat index into 2D index within shape."""
    return (index // shape[1], index % shape[1])


@numba.njit
def _refine_reflection_positions_locally(coords, image, half_width, mask, out):
    """Locally adjust reflection positions."""
    for i, coord in enumerate(coords):
        if not mask[i]:
            continue
        region = image[
            coord[0] - half_width : coord[0] + half_width + 1,
            coord[1] - half_width : coord[1] + half_width + 1,
        ]
        if not region.sum():
            continue
        index = region.argmax()
        out[i] = np.array(_unravel_index_2d(index, region.shape)) - half_width + coord


def refine_reflection_positions_locally(coords, image, width=5):
    """Locally refine individual reflection positions based on peaks in image.

    Parameters
    ----------
    coords: (N, 2) ndarray
        ij coodinates of the reflection positions on the image.
    image: ndarray
        The image used to adjust the reflection positions. It is
        recommended to use a filtered image, eg. gaussian or laplace.
    width: int
        The side length of the local region box. For each ij the maximum
        value within a local region around this position is determined
        to be the refined position.

    Returns
    -------
    coords: (N, 2) ndarray of int
        The refined coordinates.
    """
    coords_rounded = coords.round().astype(int)
    mask = check_bounds_coords(coords_rounded, image.shape, buffer=width)
    half_width = width // 2

    out = np.copy(coords_rounded)
    _refine_reflection_positions_locally(coords_rounded, image, half_width, mask, out)
    return out


def check_bounds_coords(coords, shape, buffer=0):
    """
    Convenience function to return mask of which coords are within array
    bounds (shape).

    Parameters
    ----------
    coords: (N, ndim) ndarray
        The coords in question.
    shape: (ndim,) array-like
        The shape of the array in question.
    buffer: float
        Buffer from array edges to consider safe.

    Returns
    -------
    mask: (N,) ndarray
        True where coords is safely within bounds, False otherwise.

    """
    l1 = coords >= 0 + buffer
    l2 = coords < np.asarray(shape) - buffer
    return np.logical_and(l1, l2).all(axis=-1)


def add_floats_to_array(arr, coords, values=None):
    """

    Distribute float values around neighbouring pixels in array whilst
    maintinating center of mass. Uses Manhattan (taxicab) distances for
    center of mass calculation.

    Parameters
    ----------
    arr: ndim ndarray
        Floats will be distributed into this array.
        Array is modified in place.
    coords: (N, ndim) ndarray
        Floats to distribute into array.
    values: None or (N,)
        The weights of each coord.
        If None then each coord is assigned 1.

    """
    coords = np.atleast_2d(coords)

    if values is None:
        values = np.ones(len(coords), dtype=arr.dtype)
    else:
        values = np.asarray(values)
    if not values.ndim == 1 and len(values) == len(coords):
        raise ValueError("Values must be (N,) ndarray.")

    if coords.shape[1] == 2:
        # quick numba implementation in 2d
        _add_floats_to_array_2d(arr, coords, values=values)
    else:
        raise ValueError("not currently supported.")
        # indices_local = np.stack(
        #     np.meshgrid(*tuple(range(2) for i in range(coords.shape[1]))),
        #     axis=1,
        # )

        # for i, c in enumerate(coords):
        #     temp = 1.0 / np.prod(np.abs(indices_local - np.remainder(c, 1)), axis=-1)
        #     arr[tuple((indices_local + c.astype(int)).T)] += (
        #         values[i] * temp / temp.sum()
        #     )


@numba.njit
def _add_floats_to_array_2d(arr, coords, values):
    """

    Distribute float values around neighbouring pixels in array whilst
    maintinating center of mass. Uses Manhattan (taxicab) distances for
    center of mass calculation.

    This function uses numba to speed up the calculation but is limited
    to exactly 2D.

    Parameters
    ----------
    arr: ndim ndarray
        Floats will be distributed into this array.
        Array is modified in place.
    coords: (N, 2) ndarray
        Floats to distribute into array.
    values: (N,) arraylike
        The total value of each coord to distribute into arr.
    """
    indices_local = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])

    for i in numba.prange(len(coords)):
        temp_abs = np.abs(indices_local - np.remainder(coords[i], 1))
        temp = (1.0 - temp_abs[..., 0]) * (1.0 - temp_abs[..., 1])
        arr[
            int(coords[i, 0]) : int(coords[i, 0]) + 2,
            int(coords[i, 1]) : int(coords[i, 1]) + 2,
        ] += (
            values[i] * temp / temp.sum()
        )


def index_array_with_floats(arr, coords, mask=None, default=np.nan):
    """
    Use float coordinates to index an array. Values are computed using
    the center of mass from neighbouring pixels. This function will work
    with arr.ndim >= 2. In all cases the final two axes are indexed.

    Parameters
    ----------
    arr: ndarray
        The array to index.
    coords: (N, 2) ndarray
        The coordinates used to index the array.
    mask: (N,) ndarray or None
        Mask of valid coordinates. False for out-of-bounds coordinates.
        If None the out-of-bounds coordinates will be computed.
    default: scalar
        The default fill value for return array.

    Returns
    -------
    out: (arr.ndim[:-2], N) ndarray
        The indexed values.

    """
    assert coords.ndim == 2, "coords should be (N, 2) array."

    if mask is None:
        # filter good coords
        mask = check_bounds_coords(coords, arr.shape[-2:])
    else:
        assert isinstance(mask, np.ndarray) and len(mask) == len(
            coords
        ), "Mask should be 1d-ndarray with True for valid coordinates."

    # default values for out of bounds coordinates
    out = np.full((*arr.shape[:-2], len(coords)), default)

    if coords.dtype in (int, np.int32, np.int64):
        # no need to index using floats
        out[mask] = arr[tuple(coords[mask].T)]
    else:  # float coords
        if arr.ndim == 2:
            _index_array_with_floats_2d(arr, coords, out, mask)
        elif arr.ndim > 2:
            _index_array_with_floats_nd(arr, coords, out, mask)
        else:
            raise ValueError(f"ndim={arr.ndim} not supported.")
    return out


def _index_array_with_floats_nd(arr, coords, out, mask):
    """
    Return the center of mass interpolated values of float indices
    within an array. Uses Manhattan (taxicab) distances for center of
    mass calculation.

    This function uses numba to speed up the calculation but is there
    limited to exactly 2D.

    Parameters
    ----------
    arr: ndarray
        The array to index.
    coords: (N, ndim) ndarray
        Float coordinates used to index array.
    out: (N,) ndarray
        The values are put into this array. Must have same dtype as arr.
    mask: (N,) ndarray
        Mask is True if coords is within arr bounds, False is outside.
        If False then these coordinates will not be computed, ie. out[i]
        will remain unchanged.

    """
    indices_local = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])

    for i in range(len(mask)):
        if not mask[i]:  # if mask is False then move on
            continue
        temp_abs = np.abs(indices_local - np.remainder(coords[i], 1))
        temp = (1.0 - temp_abs[..., 0]) * (1.0 - temp_abs[..., 1])
        local = arr[
            ...,
            int(coords[i, 0]) : int(coords[i, 0]) + 2,
            int(coords[i, 1]) : int(coords[i, 1]) + 2,
        ]
        # weighted average
        out[..., i] = (local * temp).sum(axis=(-2, -1)) / temp.sum()


@numba.njit
def _index_array_with_floats_2d(arr, coords, out, mask):
    """
    Return the center of mass interpolated values of float indices
    within an array. Uses Manhattan (taxicab) distances for center of
    mass calculation.

    This function uses numba to speed up the calculation but is there
    limited to exactly 2D.

    Parameters
    ----------
    arr: ndarray
        The array to index.
    coords: (N, ndim) ndarray
        Float coordinates used to index array.
    out: (N,) ndarray
        The values are put into this array. Must have same dtype as arr.
    mask: (N,) ndarray
        Mask is True if coords is within arr bounds, False is outside.
        If False then these coordinates will not be computed, ie. out[i]
         will remain unchanged.

    """
    indices_local = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])

    for i in numba.prange(len(mask)):
        if not mask[i]:  # if mask is False then move on
            continue
        temp_abs = np.abs(indices_local - np.remainder(coords[i], 1))
        temp = (1.0 - temp_abs[..., 0]) * (1.0 - temp_abs[..., 1])
        # handle perfect integers
        local = arr[
            int(coords[i, 0]) : int(coords[i, 0]) + 2,
            int(coords[i, 1]) : int(coords[i, 1]) + 2,
        ]
        out[i] = (local * temp).sum() / temp.sum()  # weighted average


def add_poisson_noise(arr: ArrayLike, lam: float, factor: float = 1.0) -> ArrayLike:
    """
    Add Poisson noise to an array. A larger lambda value equates to more
    noise. Lambda is subtracted after the noise is applied to keep the
    original mean.

    Parameters
    ----------
    arr: ArrayLike
        Array to apply noise to.
    lam: float
        The lambda value of the Poisson distribution.
    factor: float
        Scaling factor of the input array before applying noise.
        This factor is removed before returning.

    Returns
    -------
    out: ArrayLike
        The input array with the applied noise.

    """
    return (np.random.poisson(lam + factor * arr).astype(arr.dtype) - lam) / factor


def bin_box(
    arr: ArrayLike,
    factor: int,
    axis: Union[None, int, Tuple] = None,
    dtype: Union[DTypeLike, bool] = False,
) -> ArrayLike:
    """

    Use box averaging to bin the images.

    Parameters
    ----------
    arr: ndarray
        Input array to box.
    factor: int
        The binning factor.
    axis: None, int, or tuple of ints
        Axis or axes to apply binning to.
        If None then all axes are binned.
    keep_dtype: bool or dtype
        If True then the output data type will be the same as arr.
        If False then the output type defaults to np.float.
        If dtype then this data type will be forced.

    Returns
    -------
    binned: ndarray
        The binned array.

    """

    arr = np.asarray(arr)

    if axis is None:
        axis = tuple(range(arr.ndim))
    else:
        if isinstance(axis, (int, np.integer)):
            axes = (axis,)
        else:
            assert isinstance(
                axis, (list, tuple)
            ), "axes must be either int, list, or tuple."

    axis = tuple(a if a >= 0 else a + arr.ndim for a in axis)  # handle negative indices
    assert max(axis) <= arr.ndim, "axes must be within arr.ndim."
    assert all(
        isinstance(i, (int, np.integer)) for i in axis
    ), "All axes must be integers."

    assert all(
        not arr.shape[i] % factor for i in filter(lambda x: x is not None, axis)
    ), f"array shape is not factorisable by factor {factor}."

    # should work ndim
    slices = []
    for v in product(
        range(factor), repeat=len(axis)
    ):  # calculate all slicing offsets in all dimensions
        v = iter(v)
        temp = []
        for i in range(arr.ndim):
            # add slice object if axes is specified, otherwise no slicing
            temp.append(slice(next(v), None, factor) if i in axis else slice(None))
        slices.append(tuple(temp))

    # sort output data type
    if dtype is True:
        dtype = arr.dtype
    elif dtype is False:
        dtype = None
    # otherwise assume a valid data type

    # stack the offset slices and take mean down stack axis to finish binning
    return np.stack(tuple(arr[s] for s in slices), axis=0).mean(axis=0, dtype=dtype)


def roll_by_pad_and_crop(arr: ArrayLike, shift: int, **kwargs) -> ArrayLike:
    """

    Roll an array by padding and cropping to avoid any rollover.

    Parameters
    ----------
    arr: ndarray
        Array to roll.
    shift: tuple of ints
        The shift values for each dimension. Same length as arr.ndim.
    kwargs: dict
        kwargs passed to numpy.pad.

    Returns
    -------
    out: ndarray
        The rolled array.

    """

    # don't roll array, but rather pad and crop
    pad = tuple((i if i > 0 else 0, -i if i < 0 else 0) for i in shift)
    slices = tuple(slice(None if i > 0 else -i, -i if i > 0 else None) for i in shift)
    # pad the array and then crop to create the roll
    return np.pad(arr, pad, **kwargs)[slices]
