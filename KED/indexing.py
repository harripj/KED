import numba
import numpy as np

from .rotation import generate_rotated_template
from .utils import DTYPE, _index_array_with_floats_2d


@numba.njit(parallel=True)
def _scan_azimuthal_rotations(
    coords,
    image,
    pixel_size,
    center,
    thetas,
    T,
    out,
    float_coords=False,
    norm_P=False,
    norm_T=False,
):
    """
    Calculate the correlation index for a set of rotations for a given template.

    Parameters
    ----------
    coords: (N, 2) ndarray
        The simulated diffraction template coordinates (ij) in 1/Angstrom.
    image: (P, M) ndarray
        The experimental diffraction pattern to index.
        pixel_size: float
        The pixel sizes of the image in 1/Angstrom.
    center: (2,) array-like
        The direct beam coodinates of the image (ij).
    thetas: (L,) ndarray
        The rotation values to compute.
    T: (N,) ndarray
        The Template intensity values.
    out: (L,) ndarray
        The empty array to store the correlation index values for each rotation.
    float_coords: bool
        If True then distribute vectors within array by center of mass (more accurate, but takes longer).
        Otherwise round vectors to index image.
    norm_P, norm_T: bool
        Whether to normalize the Pattern and Template intensities.

    """
    for i in numba.prange(len(thetas)):
        # rotate coords around (0, 0) then add center shift
        temp = generate_rotated_template(coords, thetas[i], pixel_size, center)
        P = np.zeros(len(temp), dtype=image.dtype)
        # make sure all coords are in image bounds
        if float_coords:
            lower_bound = 0
            upper_bound = 1
        else:
            lower_bound = -0.5 + 1e-6  # ensure rounds up
            upper_bound = 0.5 - 1e-6  # rensure ounds down
        mask = (
            (temp[..., 0] >= lower_bound)
            * (temp[..., 0] <= image.shape[0] - upper_bound)
            * (temp[..., 1] >= lower_bound)
            * (temp[..., 1] <= image.shape[1] - upper_bound)
        )
        # do heavy lifting
        if float_coords:
            _index_array_with_floats_2d(image, temp, P, mask)
        else:
            for j in numba.prange(len(temp)):
                if not mask[j]:
                    continue
                P[j] = image[int(round(temp[j, 0])), int(round(temp[j, 1]))]

        out[i] = _calculate_correlation_index(
            P[mask], T[mask], norm_P=norm_P, norm_T=norm_T
        )


@numba.njit
def _calculate_correlation_index(P, T, norm_P=True, norm_T=True, dtype=DTYPE):
    """
    Compute and normalise the correlation index.

    Parameters
    ----------
    P: (N,) ndarray
        The experimental Pattern intensities.
    T: (N,) ndarray
        The simulated Template intensities.
    norm_P, norm_T: bool
        Whether to normalise the correlation index by P and T.
    dtype: dtype
        Output datatype.

    Returns
    -------
    CI: float
        The correlation index.

    """
    norm = 1.0
    if norm_P:
        norm *= np.linalg.norm(P)
    if norm_T:
        norm *= np.linalg.norm(T)

    # in the case of simulated patterns, P may be all 0.
    # check if norm is small, and if so, set to 0
    if norm <= 1e-6:
        out = 0.0
    else:
        out = (P * T).sum(dtype=dtype) / norm
    return out


@numba.njit(parallel=True)
def round_numba(arr, out):
    # rounded = np.round(arr)
    for i in numba.prange(len(arr)):
        out[i] = int(round(arr[i]))


@numba.njit(parallel=True)
def _scan_pixel_sizes(
    coords,
    image,
    center,
    pixel_sizes,
    thetas,
    T,
    out,
    norm_P=False,
    norm_T=True,
    float_coords=False,
):
    """
    Compute the best correlation index from a range of pixel sizes.

    Parameters
    ----------
    coords: (N, 2) ndarray
        The simulated diffraction template coordinates (ij) in 1/Angstrom.
    image: (N, M) ndarray
        The experimental diffraction pattern to index.
    center: (2,) array-like
        The direct beam coodinates of the image (ij).
    pixel_sizes: array-like
        The pixel sizes to scan.
    T: (N,) ndarray
        The simulated template intensities.
    out: (N, 2) ndarray
        The calculated correlation index and optimum rotation value for each pixel size is stored in this array.
        Column 0 is correlation index and column 1 is rotation value.
    norm_P, norm_T: bool
        Whether to normalise the correlation index by P and T.
    float_coords: bool
        If True then distribute vectors within array by center of mass (more accurate, but takes longer).
        Otherwise round vectors to index image.

    """

    for i in numba.prange(len(pixel_sizes)):
        temp = np.empty(len(thetas), dtype=out.dtype)
        _scan_azimuthal_rotations(
            coords,
            image,
            pixel_sizes[i],
            center,
            thetas,
            T,
            temp,
            float_coords=float_coords,
            norm_P=norm_P,
            norm_T=norm_T,
        )

        out[i, 0] = temp.max()
        out[i, 1] = thetas[temp.argmax()]

        # # make sure coords are in bounds
        # mask = (_coords[..., 0] >= 0) * (_coords[..., 0] < shape[0]) * (_coords[..., 1] >= 0) * (_coords[..., 1] < shape[1])
        # P = np.empty(mask.sum(), dtype=image.dtype)

        # _temp = _coords[mask]

        # if float_coords:
        #     _index_floats_array_2d(_temp, image, P)
        #     # pass
        # else:
        #     for j in numba.prange(len(_temp)):
        #         P[j] = image[int(round(_temp[j, 0])), int(round(_temp[j, 1]))]

        # out[i] = _calculate_correlation_index(P, T[mask], norm_P=norm_P, norm_T=norm_T)
