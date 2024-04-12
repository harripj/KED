from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy import ndimage, signal
from skimage import draw, morphology

from .utils import DTYPE


def kernel_blurred_disk(
    radius: int, sigma: float, ndim: int = 2, dtype: DTypeLike = DTYPE
) -> NDArray:
    """
    Simulate a blurred disk kernel.

    Parameters
    ----------
    radius: scalar
        Disk radius in pixels.
    sigma: scalar
        Gaussian width in pixels.
    ndim: int
        Kernel dimensionality.
    dtype: DTypeLike
        The dtype of the calculated kernel.

    Returns
    -------
    kernel: ndarray
        The simulated kernel.
    """
    if ndim != 2:
        raise ValueError("Only ndim = 2 is currently supported.")

    # kernel size in pixels
    size = int(round(2 * (radius + 2 * sigma) + 1))
    kernel = np.zeros((size,) * ndim, dtype=dtype)

    kernel[draw.disk(((size - 1) // 2,) * ndim, radius, shape=kernel.shape)] = 1
    kernel = ndimage.gaussian_filter(kernel, sigma)

    # normalize kernel
    kernel /= kernel.sum()

    return kernel


def apply_point_spread_function(
    arr: ArrayLike,
    kernel: Union[ArrayLike, Callable],
    mode: str = "same",
    positive: bool = True,
) -> NDArray:
    """

    Apply point spread function (PSF) to an array.

    Parameters
    ----------
    arr: ndarray
        Array to apply PSF to.
    kernel: arr.ndim ndarray or float
        Point spread function. All reflections are convoluted with this
        function. If ndarray then it is a PSF kernel. If int or float
        then the template will be convoluted with a disk with this
        radius. If func then must be of form func(x: ndarray) -> ndarray
        Pass 0 for no PSF to be applied.
    mode: bool
        Passed to scipy.signal.convolve.
    positive: bool
        If True all negative values after convolution are set to 0.

    Returns
    -------
    out: ndarray
        The array with PSF applied.

    """
    is_callable = False
    # format PSF kernel
    if isinstance(kernel, (int, float, np.integer)):
        kernel = morphology.disk(kernel)
    elif isinstance(kernel, np.ndarray):
        assert kernel.ndim == arr.ndim, (
            "psf must have the same dimensionality as shape: "
            + f"{kernel.ndim} != {len(arr.shape)}."
        )
    elif callable(kernel):
        is_callable = True
    else:
        raise ValueError("psf must be either ndarray or scalar.")

    if is_callable:
        out = kernel(arr)
    else:
        out = signal.convolve(arr, kernel, mode=mode)

    if positive:
        out[out < 0] = 0

    return out
