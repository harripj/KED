import math
from typing import List, Literal, Optional, Tuple, Union

from matplotlib.pyplot import Axes
import numpy as np
from numpy.typing import ArrayLike, NDArray
from orix.quaternion import Orientation, Quaternion, Rotation
from orix.quaternion.symmetry import C1, Symmetry
from orix.vector import AxAngle
from scipy import ndimage, optimize
from scipy.spatial.transform import Rotation as spRotation
from tqdm.auto import tqdm

from .utils import generate_thetas


def format_matrix(x: ArrayLike) -> NDArray:
    """
    Make sure rotation matrix is square and at least 3d.

    Parameters
    ----------
    x: ndarray
        Rotation matrix.

    Returns
    -------
    x: ndarray
        At least 3d matrix.
    """
    x = np.asarray(x)
    if x.ndim < 2:
        raise ValueError("Array must have at least 2 dimensions.")

    dimensionality = set(x.shape[-2:])
    if len(dimensionality) != 1:
        raise ValueError(f"x is not x square array: {x.shape}")
    if list(dimensionality)[0] not in {2, 3}:
        raise ValueError("x is not a 2d or 3d rotation matrix.")

    return x[np.newaxis] if x.ndim == 2 else x


def convert_to_scipy(x: Union[NDArray, Quaternion, spRotation]) -> spRotation:
    """
    Convert either rotation matrix or Quaternion to scipy Rotation.

    Parameters
    ----------
    x: ndarray or orix.quaternion.Quaternion
        Rotations to convert.

    Returns
    -------
    rot: scipy.spatial.transform.Rotation
        Rotation objects.
    """
    if isinstance(x, np.ndarray):
        out = Rotation.from_matrix(format_matrix(x))
    elif isinstance(x, Quaternion):
        out = Rotation.from_quat(x.data[..., [1, 2, 3, 0]])
    elif isinstance(x, spRotation):
        out = x
    else:
        raise ValueError(
            "x must be either a rotation matrix (ndarray) or Quaternion object."
        )

    return out


def convert_to_orix(x: Union[NDArray, Quaternion, spRotation]) -> Quaternion:
    """
    Convert either rotation matrix or scipy Rotation to orix Quaternion.

    Parameters
    ----------
    x: ndarray or scipy.spatial.transform.Rotation
        Rotations to convert.

    Returns
    -------
    quat: orix.quaternion.Quaternion
        Quaternion rotation object.
    """
    if isinstance(x, np.ndarray):
        out = Orientation.from_matrix(format_matrix(x))
    elif isinstance(x, spRotation):
        out = Orientation(np.atleast_2d(x.as_quat())[..., [3, 0, 1, 2]])
    elif isinstance(x, Quaternion):
        out = x
    else:
        raise ValueError(
            "x must be either a rotation matrix (ndarray) or Rotation object."
        )

    return out


def convert_to_matrix(x: Union[NDArray, Quaternion, spRotation]) -> NDArray:
    """
    Convert either scipy Rotation or orix Quaternion to rotation matrix.

    Parameters
    ----------
    x: orix.quaternion.Quaternion or scipy.spatial.transform.Rotation
        Rotations to convert.

    Returns
    -------
    mat: ndarray
        Rotation matrix.
    """
    if isinstance(x, Quaternion):
        out = x.to_matrix()
    elif isinstance(x, spRotation):
        out = x.as_matrix()
    elif isinstance(x, np.ndarray):
        out = x
    else:
        raise ValueError("x must be either a Quaternion or Rotation object.")

    return out


def rotation_between_vectors(v1: ArrayLike, v2: ArrayLike) -> Rotation:
    """Calculate the rotation that takes v1 to v2."""
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v3 = np.cross(v1, v2)
    angle = np.arcsin(np.linalg.norm(v3) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return Rotation.from_axes_angles(v3, angle)


def compute_symmetry_reduced_orientation(
    ori1: Orientation, ori2: Orientation
) -> Orientation:
    """Compute the symmetry reduced orientations of ori1 with the
    smallest disorientation angle to ori2.

    Parameters
    ----------
    ori1: orix.quaternion.Orientation
        The orientations from which the the symmetry operation with the
        smallest disorientation to ori2 will be computed. Symmetry
        property should be defined.
    ori2: (1,) orix.quaternion.Orientation
        The reference orientation. Symmetry property should be defined.

    Returns
    -------
    ori1: orix.quaternion.Orientation
        The symmetry orientation of ori1 with the smallest
        disorientation angle to ori2. Same shape as ori1.

    """
    if not all(isinstance(o, Quaternion) for o in (ori1, ori2)):
        raise TypeError("ori1 and ori2 must be orix.quaternion.Quaternion.")

    if ori2.size != 1 or ori2.ndim != 1:
        raise ValueError("ori2 must have shape (1,).")

    if ori1.symmetry == ori2.symmetry:
        symmetry = ori1.symmetry
    else:
        symmetry = ori1.symmetry.outer(ori2.symmetry).unique()

    ori1_sym = symmetry.outer(ori1)
    misorientation = ori2 * ~ori1_sym
    angles = misorientation.angle
    indices = angles.argmin(axis=0)  # minimum angle down symmetry dim.
    # index one symmetry orientation for each initial orientation
    # squeeze last redundant dimension
    out = np.take_along_axis(ori1_sym, indices[np.newaxis], axis=0).squeeze()
    return ori1.__class__(out)


def get_closest_symmetry_orientation(
    ori1: Orientation, ori2: Orientation, n: int = 1
) -> Tuple[Orientation, int]:
    """Compute the symmetry reduced orientation of ori1 which has
    smallest disorientation angle to ori2.

    Parameters
    ----------
    ori1: orix.quaternion.Orientation
        The orientations from which the single symmetry orientation with
        the smallest disorientation angle to ori2 will be computed.
    ori2: (1,) orix.quaternion.Orientation
        The reference orientation.
    symmetry: orix.quaternion.Symmetry
        The symmetry operation.
    n: int
        The number of orientations and indices to return in order of
        ascending disorientation angle.

    Returns
    -------
    ori1: (1,) orix.quaternion.Orientation
        The symmetry orientation of an orientation within ori1 which has
        the smallest disorientation angle to ori2.
    index: int
        The flattened index of this orientation within ori1.

    """
    assert ori2.size == 1, "ori2 must have shape (1,)."
    assert n > 0 and isinstance(n, (int, np.integer)), "n must be a positive integer."
    ori1s = compute_symmetry_reduced_orientation(ori1, ori2).flatten()
    misori = ori1s * ~ori2
    index_sorted = misori.angle.argsort()
    indices = index_sorted[:n]
    out = ori1s[indices]
    # below is the test for this function
    # assert np.isclose(
    #     (Orientation(out.outer(symmetry)) - ori1[index]).angle.data.min(), 0
    # )
    return out, np.squeeze(indices)


def track_orientation(
    ori: Orientation,
    orientations: List[Orientation],
    rot: Rotation,
    index: int = 0,
    tracking: Literal["previous", "expected"] = "previous",
    return_indices: bool = False,
    n: int = 1,
) -> Tuple[Orientation, List[int]]:
    """Track an orientation through the set of orientations.

    Parameters
    ----------
    ori: orix.quaternion.Orientation
        The orientation to track through the sets of orientations.
        The symmetry property of ori should be set.
    orientations: (N,)-tuple of (M,) orix.quaternion.Orientation
        The set of orientations through which ori will be tracked.
        In each of N Orientations, the closest orientation within the
        set of M will be selected. The symmetry property of orientations
        should be set.
    rot: (N-1,) orix.quaternion.Rotation
        The rotations which couple each set of Orientations to the next.
        ie. rotvecs[2] couples the orientations[2] to orientations[3].
    index: int
        The index of ori within orientations used to start the tracking.
    tracking: str
        Either "previous" or "expected". If "previous" then the last
        orientation is rotated to find the next orientation. If
        "expected" then the rotated initial orientation ori is used.
    n: int
        The number of orientations and indices to return in order of
        ascending disorientation angle.
    index: int
        The N-index of ori within orientations.

    Returns
    -------
    ori: (N,) orix.quaternion.Orientation
        The closest orientation in each set.
    indices: (N,) list
        The flattened indices of the returned orientations within their
        respective sets. Returned if return_indices is True.

    """
    if not isinstance(ori, Orientation) or ori.size != 1:
        raise ValueError("ori must ne Orientation with shape (1,).")
    if not isinstance(orientations, (list, tuple)):
        raise TypeError(
            "orientations must be (N,)-tuple of orix.quaternion.Orientation."
        )
    if not all(isinstance(o, Orientation) for o in orientations):
        raise TypeError(
            "orientations must be (N,)-tuple of orix.quaternion.Orientation."
        )

    if len(orientations) != rot.size + 1:
        raise ValueError(
            "Number of orientations is not 1 greater than number of rotvecs: "
            + f"{len(orientations)} != {rot.size} + 1."
        )

    if index < 0 or index >= len(orientations):
        raise ValueError(f"index must be within (0, {len(orientations)}).")

    tracking = tracking.lower()
    if tracking not in ("expected", "previous"):
        raise ValueError("tracking must be either 'expected' or 'previous'.")

    # setup tracked orientations
    out = [None for _ in orientations]
    indices = [None for _ in orientations]
    # find initial orientation within set
    idx = ori.angle_with(orientations[index]).argsort()[:n]
    out[index] = orientations[index][idx]
    indices[index] = idx

    fwd = range(index + 1, len(orientations))
    bwd = reversed(range(index))

    rot_chained = chain_rotations(rot, index)

    for i in fwd:
        # require best match [0] to find n next best matches
        o_prev = out[i - 1][0]
        if tracking == "previous":
            ori_rotated = ~(rot[i - 1] * ~o_prev)
        else:
            ori_rotated = ~(rot_chained[i - 1] * ~ori)
        ori_rotated.symmetry = o_prev.symmetry
        idx = ori_rotated.angle_with(orientations[i]).argsort()[:n]
        out[i] = orientations[i][idx]
        indices[i] = idx

    for i in bwd:
        # require best match [0] to find n next best matches
        o_prev = out[i + 1][0]
        if tracking == "previous":
            ori_rotated = ~(~rot[i] * ~o_prev)
        else:
            ori_rotated = ~(rot_chained[i] * ~ori)
        ori_rotated.symmetry = o_prev.symmetry
        idx = ori_rotated.angle_with(orientations[i]).argsort()[:n]
        out[i] = orientations[i][idx]
        indices[i] = idx

    out = Orientation(np.stack([o.data for o in out], axis=0)).squeeze()
    out.symmetry = ori.symmetry
    return (out, np.squeeze(indices)) if return_indices else out


def filter_tracked_components(
    indices: ArrayLike,
    component_images: ArrayLike,
    size_weighting: float = 0.25,
    displacement_weighting: float = 0.8,
    tilt_axis: Literal["x", "y"] = "y",
    transform: Optional[NDArray] = None,
    index: int = 0,
) -> NDArray:
    """Given more than one possible tracked component, use other
    parameters such as displacement and size to choose the correct
    component.

    Parameters
    ----------
    indices: (N, M) array-like
        The indices of the M possible tracked orientations within N sets.
        The best orientation is index 0 for each set.
    component_images: (N,)-tuple of 3d ndarray
        The 3d stack 2d component images for each N set.
    size_weighting, displacement_weighting: floats
        The weighting factor for the size (area) error and displacement
        (parallel to tilt axis) error. The total residual is used to
        order the possibilities and the residual is defined as:
        size_weighting * abs(size - size_ref) + displacement_weighting
        * (position - position_ref).
    tilt_axis: str
        Either 'x' or 'y'. The aligned images should not displace parallel
        to the tilt axis, so displacement along this axis is considered
        for displacement_threshold.
    transform: None or (N, 6) ndarray
        If given this is the set of image transform parameters for each
        N set.
    index: int
        The reference component of N, NB M=0.

    Returns
    -------
    indices: (N, M) array-like
        The filtered indices.
    """
    tilt_axis = tilt_axis.lower()
    if tilt_axis not in ("x", "y"):
        raise ValueError("tilt_axis must be either 'x' or 'y'.")

    assert len(component_images) == len(indices)

    out = np.empty_like(indices)

    # start with index, ie. manually chosen reference pointt
    for i in (index, *range(index + 1, len(indices)), *range(0, index)[::-1]):
        comp_images = [component_images[i][idx] for idx in indices[i]]
        if transform is not None:
            # comp_images = [apply_transform(transform[i], im) for im in comp_images]
            pass
        sizes = [np.count_nonzero(im) for im in comp_images]
        com = [ndimage.center_of_mass(im) for im in comp_images]
        # only take into account displacement parallel to tilt axis
        # which should be minimized
        positions = [c[0] if tilt_axis == "y" else c[1] for c in com]

        if i == index:
            size_ref = sizes[0]
            position_ref = positions[0]

        sizes = np.array(sizes)
        positions = np.array(positions)

        weight = size_weighting * np.abs(
            sizes - size_ref
        ) + displacement_weighting * np.abs(positions - position_ref)
        out[i] = indices[i][np.argsort(weight)]
        # order = [indices[i][0]]
        # for j in range(1, len(comp_images)):
        #     # if position and size are much better, then change order
        #     cond1 = abs(positions[j]- position_ref) < displacement_threshold * abs(positions[0] - position_ref)
        #     cond2 = abs(sizes[j] - size_ref) < size_threshold * abs(sizes[0] - size_ref)
        #     if cond1 and cond2:
        #         order.insert(0, indices[i][j])
        #     else:
        #         order.append(indices[i][j])

        # out[i] = order

    return out


def chain_rotations(
    rot: Rotation, index: int = 0, insert_identity: bool = False
) -> Rotation:
    """
    Apply a set of orientations in order. It is assumed that the
    rotations are applied in sequence from index 0. It is also assumed
    that each rotation links one index to the next (chain).

    Parameters
    ----------
    rot: (N,) orix.quaternion.Rotation
        The chained rotations ordered.
        ie. rot[0] links 0 -> 1, rot[1] links 1 -> 2 etc.
    index: int
        The index within the rot list from where the first rotation is
        applied. If greater than 0 some rotations are therefore inversed
        before calculation. The rotations supplied to rot should not be
        previously inverted.
    insert_identity: bool
        If True the identity rotation is inserted at index. The
        resulting returned shape therefore becomes (N+1,).

    Returns
    -------
    rot_chained: (N,) or (N+1,) orix.quaternion.Rotation
        The chained rotations.
    """
    rot = convert_to_orix(rot)
    if not rot.ndim == 1:
        raise ValueError("Rot should be 1-dimensional.")

    bwd = []
    # go backwards
    # 0 initial rotation to start the chain
    temp = Orientation.identity()
    for i in reversed(range(index)):
        # apply inverse rotvec backwards
        temp = ~rot[i] * temp
        bwd.append(temp)
    # convert to forward order
    bwd = bwd[::-1]
    if insert_identity:
        bwd.append(Rotation.identity())

    fwd = []
    # 0 initial rotation to start the chain
    temp = Orientation.identity()
    for i in range(index, rot.size):
        temp = rot[i] * temp
        fwd.append(temp)

    # stack the rotations correctly and create new Rotation instance
    data = (*[b.data for b in bwd], *[f.data for f in fwd])
    return Rotation(np.concatenate(data))


def apply_chained_rotations(
    rot: Rotation, ori: Orientation, index: int = 0
) -> Orientation:
    """
    Apply a set of orientations in order to a given orientation.
    It is assumed that the rotations are applied in sequence from index
    0. It is also assumed that each rotation links one index to the next
    (chain).

    Parameters
    ----------
    rot: (N,) orix.quaternion.Orientation
        The chained rotations ordered.
        ie. rot[0] links 0 -> 1, rot[1] links 1 -> 2 etc.
    ori: orix.quaternion.Orientation
        The orientation to apply the chained rotations to.
    index: int
        The index within the rot list from where the first rotation is
        applied. If greater than 0 some rotations are therefore inversed
        before calculation. The rotations supplied to rot should not be
        previously inverted.

    Returns
    -------
    ori_rotated: (N+1,) orix.quaternion.Orientation
        The new orientations. As it is assumed that each rot links two
        orientations, N+1 orientations are returned. ori is included in
        this set at ref_index.
    """
    rot = chain_rotations(rot, index=index)
    before = [r for r in rot[:index]]
    after = [r for r in rot[index:]]

    # reference orientation has no rotation applied to it
    data = (
        *[b.data for b in before],
        Orientation.identity().data,
        *[a.data for a in after],
    )
    rot_chained = Rotation(np.concatenate(data))
    # apply the rotations
    return ~(rot_chained * ~ori)


def rotation_axis_from_theta(theta: ArrayLike, z: ArrayLike = 0) -> AxAngle:
    """
    Convenience function to return normalized rotation axis from theta
    and z.

    Parameters
    ----------
    theta: (N,) array-like or scalar
        Theta values in radians.
    z: (N,) array-like or scalar
        z-component of the axis.

    Returns
    -------
    AxAngle: (N, 3) orix.vector.AxAngle
        The rotation axis (x, y, z).
    """
    axis = np.column_stack((np.cos(theta), np.sin(theta), np.full_like(theta, z)))
    # normalize axis
    return AxAngle(axis).unit


def compute_rotated_crystal_disorientation_matrix(
    rot: Rotation, ori1: Orientation, ori2: Orientation, match: bool = False
) -> Union[NDArray, Tuple[NDArray, Tuple[NDArray, NDArray]]]:
    """Compute the disorientation matrix in the crystal reference frame
    after rotation in the lab reference frame.

    Parameters
    ----------
    rot: orix.quaternion.Rotation
        The rotation in the lab reference frame.
    ori1, ori2: (N,) and (M,) orix.quaternion.Orientation
        The two sets of orientations in the crystal reference frame with
        their defined symmetries.
    match: bool
        If True then the minimum 1-to-1 matched disorientation residual
        is computed. In this case the matching indices (i, j) which
        return the optimum match from the distance matrix are also
        returned.

    Returns
    -------
    dist: (N, M) np.ndarray
        The disorientation matrix between the two sets of orientations.
    i, j: (N,) np.ndarray, optional
        The matching indices, returned if match is True.
    """
    if not isinstance(rot, Rotation):
        raise ValueError("Rot must be orix.quaternion.Rotation.")
    if not all(isinstance(o, Orientation) for o in (ori1, ori2)):
        raise ValueError("Ori must be orix.quaternion.Orientation.")
    # bring ori1 to lab reference frame, rotate, then back to
    # crystal reference frame
    ori1r = ~(rot * ~ori1)
    dist = ori1r.angle_with_outer(ori2)
    if match:
        # compute optimum map
        rows, cols = optimize.linear_sum_assignment(dist)
        return dist, (rows, cols)
    else:
        return dist


def scan_tilt_axis(
    angle: float,
    ori1: Orientation,
    ori2: Orientation,
    z: float = 0.0,
    n: int = 360,
    max_angle: Optional[float] = None,
    refine: bool = False,
    refine_max_angle: float = 2.0,
    ax: Optional[Axes] = None,
    return_residuals: bool = False,
    **kwargs,
) -> Union[Rotation, Tuple[Rotation, NDArray]]:
    """
    Scan lab rotation axes that maps ori1 onto ori2.

    Ori1 and ori2 are crystal orientations in the crystal reference
    frame. Ori1 will be brought to the lab reference frame, rotated by
    angle, and then returned to the crystal reference frame to evaluate
    misorientation with ori1.

    After transformation the residuals between the two (unordered) sets
    of points are computed. The minimum residual is a good estimation of
    the tilt axis.

    Typically the angle between the two sets of orientations is known
    and should be defined here. Also, in electron microscopy
    experiments, the z-axis of the rotation is typically 0, but can be
    adjusted here.

    Parameters
    ----------
    angle: float
        Angle between the two sets of orientations in degrees.
    ori1, ori2: orix.quaternion.Quaternion
        The two sets of orientations in the crystal reference frame.
        The rotation is applied to ori1. These symmetry attributes of
        these orientations should already be set.
    z: float
        The z-component of the approximate rotation vector, this is
        typically 0.
    n: int
        The number of angles to probe over the range [0..2*pi].
    ax: None or mpl.Axes
        If provided then the residuals will be plotted on these axes.
    max_angle: float or None
        Only pairs with misorientation less than this angle (in degrees)
        are considered. No filtering is performed if None.
    refine: bool
        If True then the optimial axis is used as the initial guess for
        refinement and the refined vector is returned.
    refine_max_angle: scalar
        Only pairs with misorientation less than this angle are used for
        refinement. Each set of orientations may be incomplete, so this
        filtering step is provided to limit skews from false pairs.
    return_residuals: bool
        If True the residuals of the scan are also returned.
    kwargs:
        Passed to scipy.optimize.minimize.

    Returns
    -------
    rotvec: orix.quaternion.Rotation
        The optimum rotation vector that maps ori1 to ori2 from the scan.

    Notes
    -----
    The scanned angles can be generated as ```thetas = generate_thetas(n)```.
    """
    # generate rotation vectors
    thetas = generate_thetas(n)
    # create rotation axis (rotvec)
    axis = rotation_axis_from_theta(thetas, z).data * np.deg2rad(angle)

    # scan rotation vectors
    residuals = np.empty_like(thetas)
    for i, a in enumerate(tqdm(axis)):
        dist, (rows, cols) = compute_rotated_crystal_disorientation_matrix(
            rotvec_to_rotation(a), ori1, ori2, match=True
        )
        # get residuals from best matches
        residuals[i] = dist[rows, cols].mean()

    if isinstance(ax, Axes):
        ax.plot(np.rad2deg(thetas), residuals)
        ax.set_xlabel("Angle/ deg.")
        ax.set_ylabel("Residuals")
    elif hasattr(ax, "plot"):
        ax.plot(np.rad2deg(thetas), residuals)

    out = axis[residuals.argmin()]

    # TODO: this needs fixing
    if refine:
        # calculate the cost matrix once for max_angle filtering
        cost = compute_rotated_crystal_disorientation_matrix(
            rotvec_to_rotation(out), ori1, ori2, match=False
        )
        mask = cost.min(axis=-1) <= np.deg2rad(refine_max_angle)
        print(
            f"{np.count_nonzero(mask)} orientations less than {refine_max_angle} "
            + "deg. misorientation used for refinement."
        )
        print(f"Initial axis: {np.round(out, 3)}")

        def mean_rotated_crystal_disorientation(rv, o1, o2):
            dist, (rows, cols) = compute_rotated_crystal_disorientation_matrix(
                rotvec_to_rotation(rv), o1, o2, match=True
            )
            return dist[rows, cols].mean()

        res = optimize.minimize(
            mean_rotated_crystal_disorientation,
            out,
            args=(ori1[mask], ori2),
            **kwargs,
        )
        out = res.x
        print(f"Refined axis: {np.round(out, 3)}")

    rot = rotvec_to_rotation(out)
    return (rot, residuals) if return_residuals else rot


def scan_rotation_angles(
    angles: ArrayLike,
    theta: ArrayLike,
    ori1: Orientation,
    ori2: Orientation,
    z: float = 0.0,
    max_angle: Optional[float] = None,
    ax: Optional[Axes] = None,
) -> Rotation:
    """
    Scan rotation rotation axes that maps ori1 onto ori2.

    After transformation the residuals between the two (unordered) set
    of points are computed. The minimum residual is a good estimation of
    the tilt axis.

    Typically the angle between the two sets of orientations is known
    and should be defined here. Also, in electron microscopy
    experiments, the z-axis of the rotation is typically 0, but can be
    adjusted here.

    Parameters
    ----------
    angles: array-like
        Rotation angles between the two sets of orientations to scan in
        degrees.
    theta: array-like or scalar
        The in-plane rotation angle(s) in degrees. If array-like then
        angles will be tested for each theta.
    ori1, ori2: orix.quaternion.Orientation
        The two sets of orientations with symmetries.
        The rotation is applied to ori1.
    z: float
        The z-component of the approximate rotation vector, this is
        typically 0.
    ax: None or mpl.Axes
        If provided then the residuals will be plotted on these axes.
    max_angle: float or None
        Only pairs with misorientation less than this angle (in degrees)
        are considered. No filtering is performed if None.

    Returns
    -------
    rot: orix.quaternion.Rotation
        The optimum rotation that maps ori1 to ori2.

    Notes
    -----
    The scanned angles can be generated as ```thetas = np.deg2rad(np.arange(n))```.
    """
    angles = np.atleast_1d(angles)
    angles_rad = np.deg2rad(angles)
    theta = np.deg2rad(np.atleast_1d(theta))

    if angles.ndim > 1:
        raise ValueError("Angles is not 1d.")
    if theta.ndim > 1:
        raise ValueError("Theta is not 1d.")

    # output shape
    shape = (angles.size, theta.size)
    residuals = np.empty(shape)

    for i, j in tqdm(np.ndindex(shape), total=math.prod(shape)):
        # create rotation axis (rotvec)
        axis = rotation_axis_from_theta(theta[j], z) * angles_rad[i]
        assert axis.size == 1, "AxAngle should have shape = (1,)."
        dist, (rows, cols) = compute_rotated_crystal_disorientation_matrix(
            rotvec_to_rotation(axis), ori1, ori2, match=True
        )
        residuals[i, j] = dist[rows, cols].mean()

    if isinstance(ax, Axes):
        for j in range(len(theta)):
            ax.plot(
                angles,
                residuals[:, j],
                label=f"$\\theta={round(np.rad2deg(theta[j]), 1)}\degree$",
            )

        ax.set_xlabel("Coupling angle/ deg.")
        ax.set_ylabel("Residuals")
        ax.legend()
    elif hasattr(ax, "plot"):
        for j in range(len(theta)):
            ax.plot(np.full(len(angles), theta[j]), residuals[:, j])

    i, j = np.unravel_index(residuals.argmin(), shape)
    rotvec = rotation_axis_from_theta(theta[j], z) * np.deg2rad(angles[i])
    assert rotvec.size == 1, "AxAngle should have shape = (1,)."

    return rotvec_to_rotation(rotvec)


def compute_angular_distance_matrix(
    ori1: Orientation,
    ori2: Orientation,
    rotvec: Optional[Rotation] = None,
    symmetry: Symmetry = C1,
) -> NDArray:
    """
    Compute the angular distance (cost) matrix between two sets of
    rotations.

    Parameters
    ----------
    ori1: Rotation or (N, 3, 3) ndarray
        These orientations are rotated by rotvec.
    ori2: Rotation or (N, 3, 3) ndarray
        The reference orientations.
        The residuals are calculated between ori2 and rotated ori1.
    rotvec: (3,) ndarray or None
        Rotation vector components. The unit vector direction is the
        rotation axis and its length is the rotation amount in radians.
        This rotation is applied to ori1. Not used if None.
    symmetry: orix.quaternion.Symmetry
        The crystal symmetry. Default is C1 (identity).

    Returns
    -------
    cost: ndarray
        The computed cost matrix. Values are in radians.
        Each value is the smallest angle between pairs considering all
        symmetries.

    """
    # format input
    ori1 = convert_to_orix(ori1)
    ori2 = convert_to_orix(ori2)

    # apply any symmetries to ori1
    if not isinstance(symmetry, symmetry.Symmetry):
        raise TypeError("symmetry must be orix.quaternion.Symmetry.")

    if rotvec is not None:
        rotvec = rotvec_to_rotation(rotvec)
        ori1 = rotvec * ori1

    ori1 = ori1.outer(symmetry)
    # compute angular distance matrix
    # take minimum over symmetry axis, ie. last axis of ori1
    misori = ori2.outer(~ori1)
    dist = misori.angle.data.min(axis=-1)

    # linear_sum_assignment only works in 2d cost array case
    if dist.ndim < 2:
        raise ValueError(f"This should not be possible: dist.ndim = {dist.ndim}.")
    elif dist.ndim > 3:
        raise ValueError("Only 2 or 3D cost arrays are supported currently.")

    return dist.T


def aligned_orientations_residuals(
    rotvec: Union[ArrayLike, Rotation],
    ori1: Orientation,
    ori2: Orientation,
    symmetry: Symmetry = C1,
    max_angle: Optional[float] = None,
    return_indices: bool = False,
    return_cost: bool = False,
) -> Union[
    float,
    Tuple[float, NDArray],
    Tuple[float, Tuple[NDArray, NDArray], NDArray],
]:
    """
    Compute the residuals between a set of orientations after rotation.
    Use this objective function with scipy.optimize.minimize.

    This function uses scipy.optimize.linea_sum_assignment with the
    angular cost matrix to find the optimal mapping. ori1 and ori2 do
    not need to be complete sets.

    Parameters
    ----------
    rotvec: (3,) ndarray or orix.quaternion.Rotation
        Rotation vector. This rotation is applied to ori1.
    ori1: orix.quaternion.Orientation or (N, 3, 3) ndarray
        These orientations are rotated by rotvec.
    ori2: orix.quaternion.Orientation or (N, 3, 3) ndarray
        The reference orientations.
        The residuals are calculated between ori2 and rotated ori1.
    symmetry: orix.quaternion.Symmetry
        The crystal symmetry. Default is C1 (identity).
    max_angle: float or None
        Only pairs with misorientation less than this angle (in degrees
        are considered. No filtering is performed if None.
    return_indices: bool
        If True the pairing indices are returned.
    return_cost: bool
        If True the cost matrix is also returned.

    Returns
    -------
    res: float
        The calculated angular residuals.

    """
    # format input
    rotvec = rotvec_to_rotation(rotvec)
    ori1 = convert_to_orix(ori1)
    ori2 = convert_to_orix(ori2)

    dist = compute_angular_distance_matrix(ori1, ori2, rotvec=rotvec, symmetry=symmetry)

    # linear_sum_assignment only works in 2d cost array case
    if dist.ndim < 2:
        raise ValueError("This should not be possible.")
    elif dist.ndim == 2:
        pass  # normal operation in 2d case
    elif dist.ndim == 3:
        # concatenate last two dimensions... why? i forget
        dist = dist.reshape(len(dist), -1)
    else:
        raise ValueError("Only 2 or 3D cost arrays are supported currently.")

    if max_angle is not None:
        dist[dist > np.deg2rad(max_angle)] = np.inf

        # remove any data with no possible matches given the filtering, ie. axis 0
        idx = np.nonzero(np.count_nonzero(np.logical_not(np.isinf(dist)), axis=1))[0]

        # compute new cost matrix
        dist = compute_angular_distance_matrix(
            ori1[idx], ori2, rotvec=rotvec, symmetry=symmetry
        )

    # compute optimum map
    rows, cols = optimize.linear_sum_assignment(dist)
    # get residuals from best matches
    res = dist[rows, cols]

    if max_angle is not None:
        # the assignment is no longer inline with ori, ie. they have
        # different lengths due to filtering so return to the original
        # index before filtering so indices in rows match with input ori1
        rows = idx

    # filter by maximum allowed angle and return mean cost
    if False:  # TODO: possibly remove this in the future
        # add while loop to stop empty mask errors
        i, factor = 1, 1.2
        while not (mask := res <= np.deg2rad(max_angle) * i * factor).sum():
            i += 1
        out = res[mask].mean()
    else:
        # return mean of misorientations between matched components
        out = res.mean()

    if return_cost or return_indices:
        out = (out,)
        if return_indices:
            out = out + ((rows, cols),)
        if return_cost:
            out = out + (dist,)

    return out


def rotvec_to_rotation(rv: Union[ArrayLike, AxAngle, Rotation]) -> Rotation:
    """Convert rotation vector (scipy, array-like, or orix) to
    orix.quaternion.Rotation."""
    if isinstance(rv, (list, tuple, np.ndarray)):
        rv = np.asarray(rv)
        if rv.shape[-1] != 3:
            raise ValueError("rv as ndarray must have shape (N, 3).")
        rv = AxAngle(rv)

    if isinstance(rv, AxAngle):
        ori = Rotation.from_axes_angles(rv, rv.angle)
    else:
        ori = convert_to_orix(rv)
    return ori


def calculate_grain_boundary_map(
    ori: Orientation, threshold: float = 5.0, symmetry: Symmetry = C1, shift: int = 1
) -> NDArray[np.bool_]:
    """Calculate grain boundary map based on disorientation threshold
    (in degrees)."""

    ori_S = ori.outer(symmetry)
    out = np.zeros(ori.shape, dtype=bool)
    threshold_rad = np.deg2rad(threshold)

    if not isinstance(shift, (list, tuple)):
        shift = (shift,)
    if not all(isinstance(i, (int, np.integer)) for i in shift):
        raise TypeError(f"Shift must be int or iterable of ints.")

    for axis in range(ori.ndim):
        for s in shift:
            temp = Orientation(np.roll(ori.data, s, axis=axis))
            angles = (ori_S * ~temp).angle.data.min(axis=-1)
            out[angles > threshold_rad] = 1

    return out
