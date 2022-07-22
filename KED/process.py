from collections.abc import Callable
import itertools
import logging
import math

import h5py
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import feature
from tqdm import tqdm

from .io import get_frame_key
from .utils import (
    DTYPE,
    check_bounds_coords,
    index_array_with_floats,
    roll_by_pad_and_crop,
)


def virtual_reconstruction(
    data,
    coords,
    template_intensities=None,
    normP=False,
    normT=False,
    sum=True,
    fn=None,
    dtype=DTYPE,
):
    """
    Compute a Frozen Template Virtual Reconstruction.

    Parameters
    ----------
    data: ndarray or generator
        Data may be 2d (individual frame) or more, eg. 4d.
        The last two axes are assumed to be the image axes.
    coords: (N, 2) ndarray
        The pixel coordinates of the template, can be floats.
    template_intensities: (N,) ndarray or None
        The expected template_intensities.
        If None then 1 is used for each sample.
    normP, normT: bool
        Normalize VR by Pattern and Template intensities respectively.
    sum: bool
        If True then the reconstructed intensity from each aperture is
        summed, otherwise the individual template apertures are returned
        individually in an array.
    fn: None or Callable
        Function to apply to each frame before reconstruction.
        If None the raw data is used.
        If provided it must be of the form x = fn(x).
        If provided calculation will be done in a loop.
    dtype: dtype
        The dtype of the calculation.

    Returns
    -------
    VR: ndarray
        The calculated VR reconstruction.

    """
    if isinstance(data, np.ndarray):
        assert data.ndim >= 2, "data must be at least 2d."
        _iterate = False
        _generator = False
    elif isinstance(data, (list, tuple)):
        initial = data[0]
        assert initial.ndim == 2, "data must be at least 2d."
        _iterate = True
        _generator = False
    # assume generator
    # Itertools.chain does not pass GeneratorType, hence hasattr
    elif hasattr(data, "__iter__"):
        initial = next(data)
        assert (
            isinstance(initial, np.ndarray) and initial.ndim == 2
        ), "Each iteration must yield 2d ndarray."
        _iterate = True
        _generator = True
    else:
        raise ValueError("data is not ndarray nor iterable.")

    if fn is None:
        fn = lambda x: x
    elif isinstance(fn, Callable):
        _test = initial if _iterate else data[0]
        assert (
            _test.ndim == fn(_test).ndim
        ), "Function does not appear to produce an image."
        _iterate = True

    # should always be 2, sanity check
    frame_shape = initial.shape if _iterate else data.shape[-2:]
    if len(frame_shape) != 2:
        raise ValueError("Only 2d images are currently supported.")

    # format coords array
    coords = np.asarray(coords)
    if coords.ndim < 2:
        raise ValueError("coords should have at least two dimensions.")
    if coords.shape[-1] != len(frame_shape):
        raise ValueError(
            f"Final dimension of coords with shape: {coords.shape} "
            + f"should be {len(frame_shape)}."
        )

    if coords.ndim == 2:
        coords_shape = len(coords)
    elif coords.ndim > 2:
        # each coord N is actually an (*M, *P, 2) array of coords
        # or a disk
        coords_shape = len(coords)
        # multiple coords per point, eg. disk (si, sj, 2, N)
        coords = coords.reshape(-1, 2)  # (N, si, sj, 2) -> (N * si * sj, 2)
    else:
        raise Exception("This should not happen.")

    # check coords bounds for indexing
    mask = check_bounds_coords(coords, frame_shape)

    if _generator:
        # add initial back to front of generator
        data = itertools.chain((initial,), data)

    fill = 0  # invalid points contribute 0 intensity
    if _iterate:
        temp = [
            index_array_with_floats(fn(d), coords, mask=mask, default=fill)
            for d in data
        ]
        intensities = np.stack(temp, axis=0)
        count = len(temp)
    else:
        intensities = index_array_with_floats(data, coords, mask=mask, default=fill)
        count = math.prod(data.shape[:-2])

    if math.prod(intensities.shape) == coords_shape * count:
        # 2d case
        pass
    else:
        intensities = (
            np.stack(np.split(intensities, coords_shape, axis=1)).sum(axis=-1).T
        )

    if template_intensities is None:
        template_intensities = np.ones(intensities.shape[-1], dtype=dtype)

    out = intensities * template_intensities
    if sum:
        out = out.sum(axis=-1)

    norm = 1
    if normP:
        norm *= np.sqrt(np.square(intensities)).sum()
    if normT:
        norm *= np.sqrt(np.square(template_intensities)).sum()

    return out / norm


def apply_mask(image, mask, center=None, _sum=True, normalize=False):
    """
    Return the intensities of the frame under the mask.

    Parameters
    ----------
    image: (M, N,) ndarray
        Image to analyse.
    mask: (M, N,) ndarray, dtype bool
        Mask to overlay.
    center: None or tuple
        Center of the mask. If None the image center will be used.
    _sum: bool
        If True the the sum of the image intensities under the mask is returned.
        Otherwise the array of intensities.
    normalize: bool
        If True the output is divided by the weight of the mask.

    Returns
    -------
    intensities: ndarray or float
        If _sum is True the the sum of the image intensities under the mask is returned.
        Otherwise the array of intensities.

    """
    assert image.shape == mask.shape, "frame and mask shapes do not match."

    image_center = np.array(image.shape) // 2

    if center is None:
        center = image_center

    # roll mask by difference between center and frame_center
    difference = center - image_center

    # apply mask, if float then multiply, if bool then False values become 0 anyway
    out = image * roll_by_pad_and_crop(mask, difference)

    if _sum:
        out = out.sum()
    if normalize:
        out = out / mask.sum()
    return out


# def apply_function_to_frame(fname, func, frames, attrs=None, **kwargs):
#     '''

#     Apply a function to many frames in hdf5 file produced by TVIPSconverter.

#     Parameters
#     ----------
#     fname: str or Path
#         Path to .hdf5 file.
#     function: callable or iterable of callables
#         Function to apply to each frame (np.array).
#         Should be of the form function(array, *args, **kwargs).
#     frames: array-like of int
#         Frames number to apply function to.
#     attrs: dict
#         Use to pass soem frame attributes to func.
#         attrs must be of the form func_arg: frame_attrs_key.
#         eg. 'center' (func arg) : 'Center location' (frame attr).
#     kwargs: dict or None
#         Passed to func.

#     Returns
#     -------
#     out: list
#         List of results of function applied to frames.

#     '''

#     # if not isinstance(functions, (list, tuple)):
#     #     functions = (functions,)

#     # if functions_kwargs is None:
#     #     functions_kwargs = tuple(dict() for i in functions)
#     # if not isinstance(functions_kwargs, (list, tuple)):
#     #     functions_kwargs = (functions_kwargs,)

#     # if use_attrs is None:
#     #     use_attrs = tuple(dict() for i in functions)
#     # if not isinstance(use_attrs, (list, tuple)):
#     #     use_attrs = (use_attrs,)

#     # make sure all functions are callable
#     # assert all(isinstance(i, Callable) for i in functions), 'All functions must be callable.'
#     assert isinstance(func, Callable), 'Function must be callable.'
#     # assert len(functions) == len(functions_kwargs), 'functions_kwargs must be None or defined for each function.'

#     if attrs is None:
#         attrs = dict()

#     # check whether frames is sorted, raise user warning if not
#     def test_sorted(l):
#         for i, l1 in enumerate(l[1:]):
#             if l1 < l[i]:
#                 return False
#         return True

#     if not test_sorted(frames):
#         logging.warning('Be aware that frames is not sorted. Output will maintain index with frames.')

#     out = []

#     with h5py.File(fname, 'r') as h5:
#         group = h5['ImageStream']

#         for i, n in enumerate(tqdm(frames, desc='Frame progress')):
#             key = get_frame_key(n)
#             # apply fn and append result
#             arr = np.asarray(group[key])

#             _kwargs = kwargs.copy()
#             # attrs input has fmt (arg: attrs key)
#             for arg, attrs_key in attrs.items():
#                 _kwargs[arg] = group[key].attrs[attrs_key]

#             result = func(arr, **_kwargs)
#             out.append(result)

#     return out


def enhance_peaks(image, sigma=2.0, clip=0, factor=-1.0, dtype=DTYPE):
    """
    Use a Laplacian of Gaussian filter and clipping to highlight peaks
    in an image. `image.dtype` should be either `float32` of `float64`.

    Parameters
    ----------
    image: ndarray
        The input image to transform.
    sigma: float
        The width of the LoG filter.
    clip: float
        The lower limit clip.
    factor: float
        Scaling factor applied after transformation.
        -1.0 is used to return positive peak intensities after transform.
    dtype: DTypeLike
        The output datatype.

    Returns
    -------
    out: ndarray
        The transformed image.

    """
    return (factor * ndimage.gaussian_laplace(image, sigma, output=dtype)).clip(clip)


def find_unique_peaks(
    image1,
    image2,
    sigma=2.0,
    threshold_rel=0.2,
    num_peaks=None,
    clip=0,
):
    """
    Find unique peaks in `image1` which are not present in `image2`.

    Unique peaks are calculated in image1 relative to image2.
    A Laplacian of Gaussian filter is used to highlight relevant peaks.
    Peaks are calculated using skimage.feature.peak_local_max.
    The calculation uses image differences and kNN search.

    Parameters
    ----------
    image1, image2: ndarray
        The images in question.
    sigma: float
        The Gaussian width of the transformation.
        2 * sigma + 1 is used as min_distance in
        skimage.feature.peak_local_max.
        Also as the distance threshold to determine isolated peaks in
        the NN search.
    threshold_rel: float
        The relative threshold for skimage.feature.peak_local_max.
    num_peaks: None or int
        The maximum number of peaks to select.
        None indicates all peaks.
    clip: float
        The lower clip limit value after transformation.

    Returns
    -------
    peaks: ndarray
        The calculated unique peak coordinates in image1.

    """
    # calculate difference image to highlight peaks in frame1
    diff = image1 - image2

    # transform images and get peaks
    image2_transformed = enhance_peaks(image2, sigma=sigma, clip=clip)
    diff_transformed = enhance_peaks(diff, sigma=sigma, clip=clip)

    # find peaks in image2 and difference image
    peaks_diff = feature.peak_local_max(
        diff_transformed, min_distance=int(sigma), threshold_rel=threshold_rel
    )
    peaks2 = feature.peak_local_max(
        image2_transformed, min_distance=int(sigma), threshold_rel=threshold_rel
    )

    # create a NN tree to find isolated peaks in diff
    tree = cKDTree(peaks2)
    # get isolated peaks between the two sets
    dist, index = tree.query(peaks_diff, k=1)
    mask = (
        dist >= 2 * sigma + 1
    )  # peaks must be at least 2 sigma apart (peak_local_max)
    peaks_isolated = peaks_diff[mask]
    # sort by intensity
    intensity = diff[tuple(peaks_isolated.T)]
    peaks_isolated = peaks_isolated[np.argsort(intensity)[::-1]]

    if num_peaks is not None:
        if num_peaks < 1:
            raise ValueError("num_peaks should be integer of value 1 or higher.")
        peaks_isolated = peaks_isolated[:num_peaks]

    return peaks_isolated


def apply_function_to_frame(fname, func, frames, attrs=None, **kwargs):
    """
    Apply a function to many frames in hdf5 file produced by TVIPSconverter.

    Parameters
    ----------
    fname: str or Path
        Path to .hdf5 file.
    function: callable or iterable of callables
        Function to apply to each frame (np.array).
        Should be of the form function(array, **kwargs).
    frames: array-like of int
        Frames number to apply function to.
    attrs: dict
        Use to pass some frame attributes to func.
        attrs must be of the form func_arg: frame_attrs_key.
        eg. 'center' (func arg) : 'Center location' (frame attr).
    kwargs: dict or None
        Passed to func.
        If more than one (n) funcs provided, there must be n values for each kwarg.
        ie. possibly tuple of repeated values.

    Returns
    -------
    out: list
        List of results of each function applied to frames.

    """

    if not isinstance(func, (list, tuple)):
        func = (func,)

    if attrs is None:
        attrs = tuple(dict() for i in func) * len(func)
    if not isinstance(attrs, (list, tuple)):
        attrs = (attrs,) * len(func)

    # make sure all functions are callable
    assert all(isinstance(i, Callable) for i in func), "All functions must be callable."
    # assert isinstance(func, Callable), 'Function must be callable.'
    assert len(func) == len(attrs), "attrs must be None or defined for each function."

    if len(func) > 1:
        assert all(
            len(i) == len(func) for i in kwargs.values()
        ), "kwargs must be defined for each function, ie. each kwarg must be iterable of same length as fn."

    # check whether frames is sorted, raise user warning if not
    def test_sorted(l):
        for i, l1 in enumerate(l[1:]):
            if l1 < l[i]:
                return False
        return True

    if not test_sorted(frames):
        logging.warning(
            "Be aware that frames is not sorted. Output will maintain index with frames."
        )

    out = []

    with h5py.File(fname, "r") as h5:
        group = h5["ImageStream"]

        for i, n in enumerate(tqdm(frames, desc="Frame progress")):
            key = get_frame_key(n)
            # apply fn and append result
            arr = np.asarray(group[key])

            results = []
            for j, (fn, _attrs) in enumerate(zip(func, attrs)):
                kw = dict((dkey, dval[j]) for dkey, dval in kwargs.items())
                # attrs input has fmt (arg: attrs key)
                for arg, attrs_key in _attrs.items():
                    kw[arg] = group[key].attrs[attrs_key]

                results.append(fn(arr, **kw))

            # if only one fn just get initial result (not as list)
            if len(results) == 1:
                out.append(results[0])
            else:
                out.append(results)

    return out


# def apply_mask_to_frame(fname, mask, frames, attrs=dict(center='Center location'), **kwargs):
#     '''

#     Wrapper function that applies the fn apply_mask to every frame.

#     Parameters
#     ----------
#     fname: str or Path
#         Path to .hdf5 file.
#     mask: (M, N) ndarray
#         Mask to overlay onto
#         Function to apply to each frame (np.array).
#         Should be of the form function(array, *args, **kwargs).
#     frames: array-like of int
#         Frames number to apply function to.
#     attrs: dict
#         Use to pass soem frame attributes to func.
#         attrs must be of the form func_arg: frame_attrs_key.
#         eg. 'center' (func arg) : 'Center location' (frame attr).
#     kwargs: dict or None
#         Passed to func.

#     Returns
#     -------
#     out: list
#         List of results of function applied to frames.

#     '''

#     return apply_function_to_frame(fname, apply_mask, frames, attrs=attrs, mask=mask, **kwargs)


def apply_mask_to_frame(
    fname, mask, frames, attrs=dict(center="Center location"), **kwargs
):
    """

    Wrapper function that applies the fn apply_mask to every frame.

    Parameters
    ----------
    fname: str or Path
        Path to .hdf5 file.
    mask: (M, N) ndarray
        Mask to overlay onto
        Function to apply to each frame (np.array).
        Should be of the form function(array, *args, **kwargs).
    frames: array-like of int
        Frames number to apply function to.
    attrs: dict
        Use to pass soem frame attributes to func.
        attrs must be of the form func_arg: frame_attrs_key.
        eg. 'center' (func arg) : 'Center location' (frame attr).
    kwargs: dict or None
        Passed to func.

    Returns
    -------
    out: list
        List of results of function applied to frames.

    """

    if not isinstance(mask, (list, tuple)):
        mask = (mask,)
    if not isinstance(attrs, (list, tuple)):
        attrs = (attrs,) * len(mask)

    assert len(mask) == len(attrs), "attrs must be defined for each mask."

    return apply_function_to_frame(
        fname, (apply_mask,) * len(mask), frames, attrs=attrs, mask=mask, **kwargs
    )


def interpolate_center_location(point, scan_shape, center_locations_corner):
    """ """
    # point is scan position in question
    i, j = point
    # scan_shape
    x, y = scan_shape

    # center locations at corner of scan
    Coo, Cxo, Coy, Cxy = np.asarray(center_locations_corner)

    Cij = (
        Coo * (x - i) * (y - j)
        + Cxo * i * (y - j)
        + Coy * (x - i) * j
        + Cxy * i * j / (x * y)
    )

    return Cij


def calculate_correlation_index(
    image,
    templates,
    axis=(-2, -1),
    normalize_image=True,
    normalize_template=True,
    sqrt=True,
):
    """

    Calculate the correlation index between an image and set of templates.


    """

    templates = np.asarray(templates)
    # normally more than one template
    # but handle the case where there is just one template
    if templates.ndim == image.ndim:
        templates = templates[np.newaxis, ...]

    assert (
        image.shape == templates.shape[-len(axis) :]
    ), "image shape: {} and templates shape: {} do not match.".format(
        image.shape, templates.shape
    )

    norm = np.ones(templates.shape[0])
    if normalize_image:
        temp = np.sqrt(np.square(image).sum()) if sqrt else image.sum()
        norm = norm * temp
    if normalize_template:
        temp = (
            np.sqrt(np.square(templates).sum(axis=axis))
            if sqrt
            else templates.sum(axis=axis)
        )
        norm = norm * temp

    return (image * templates).sum(axis=axis) / norm


# # TODO:...
# def minimize_template(x, image, atoms, kwargs_generate, kwargs_project, kwargs_CI):
#     """
#     Objective function to maximize the correlation index.

#     Parameters
#     ----------
#     x: array-like
#         Euler angles (psi1, phi, psi2) in degrees.
#     image: ndarray
#         Reference image.
#     atoms: ase.Atoms
#         Needed to generate a new template.
#     kwargs_generate, kwargs_project, kwargs_CI: dicts
#         Passed to generate_diffraction_pattern, project_pattern, and calculate_correlation_index respectively.

#     Returns
#     -------
#     CI: float
#         Residual difference between two frames.

#     """

#     dp = generate_diffraction_pattern(atoms, x, **kwargs_generate)
#     template = project_pattern(dp, image.shape, **kwargs_project)

#     # maximize CI is to minimize negative CI
#     return -1.0 * calculate_correlation_index(image, template, **kwargs_CI)


# def calculate_correlation_index(
#     image,
#     templates,
#     pixel_size,
#     kind="template",
#     axis=(-2, -1),
#     normalize_image=True,
#     normalize_template=True,
#     sqrt=True,
# ):
#     """

#     Calculate the correlation index between an image and set of templates.

#     Parameters
#     ----------
#     image: (M, N) ndarray
#         The experimental image to compute.
#     templates: ([P,] M, N) ndarray or ([P,] K, image.ndim)
#         If kind='pattern' then templates should be an stack of simulated templates with the same shape as image.
#         If kind='template' then templates should be a stack of point data in array-coordinates.
#     pixel_size: float
#         Pixel size in 1/Angstroms of the image.
#     axis: int or tuple of ints
#         The axis to sum if kind='template'.
#     normalize_image, normalize_template: bool
#         If True then image and/or template intensities are normalized.
#     sqrt: bool
#         Normalise by the square root of the sum of squares.

#     Returns
#     -------
#     correlation_index: (P,) ndarray
#         The calculated correlation index for each template.

#     """
#     kinds = ("pattern", "template")
#     if kind.lower() not in kinds:
#         raise ValueError(f"kind must be one of {kinds}.")

#     # force iterable
#     if not isinstance(templates, (list, tuple, np.ndarray)):
#         templates = (templates,)

#     # perform image normalization if needed
#     if normalize_image:
#         if sqrt:
#             image = image / np.sqrt(np.square(image).sum())
#         else:
#             image = image / image.sum()

#     if kind.lower() == kinds[0]:
#         if all(isinstance(t, DiffractionPattern) for t in templates):
#             templates = np.array(tuple(t.image for t in templates))
#         else:
#             # force array dtype
#             templates = np.asarray(templates)
#         # normally more than one template
#         # but handle the case where there is just one template
#         if templates.ndim == image.ndim:
#             templates = templates[np.newaxis, ...]

#         assert (
#             image.shape == templates.shape[-len(axis) :]
#         ), "image shape: {} and templates shape: {} do not match.".format(
#             image.shape, templates.shape
#         )

#         # perform calculation
#         if normalize_template:
#             norm = (
#                 np.sqrt(np.square(templates).sum(axis=axis))
#                 if sqrt
#                 else templates.sum(axis=axis)
#             )
#         out = np.squeeze((image * templates).sum(axis=axis) / norm)

#     elif kind.lower() == kinds[1]:
#         assert all(
#             isinstance(t.__name__, "DiffractionTemplate") for t in templates
#         ), "templates should be DiffractionTemplates."

#         # if templates.ndim == 2:
#         #     templates = templates[np.newaxis, ...]
#         # # templates are not necessarily the same shape
#         # assert all(t.shape[-1] == image.ndim+1 for t in templates), f'Point template data last dimension size must equal image.ndim + 1: last axis {templates.shape} != {image.ndim} +.1.'

#         out = np.empty(len(templates), dtype=image.dtype)

#         for i, template in enumerate(templates):
#             # filter coords to be within range
#             coords = np.column_stack((template.y, template.x)) / pixel_size
#             mask = np.logical_and(
#                 np.all(coords >= 0, axis=1),
#                 np.all(coords <= tuple(s - 1 for s in image.shape), axis=1),
#             )

#             temp = np.empty(np.count_nonzero(mask), dtype=image.dtype)
#             _index_array_floats_2d(image, coords[mask], temp)

#             if normalize_template:
#                 if sqrt:
#                     temp = temp / np.sqrt(np.square(template.intensity[mask]).sum())
#                 else:
#                     temp = temp / template.intensity[mask].sum()

#             out[i] = temp.sum()

#             # fig, ax = plt.subplots()
#             # ax.matshow(image)
#             # ax.plot(coords[:,1][mask], coords[:,0][mask], 'r.')

#         # # concatenate all of the coordinates, index, and then resample
#         # coords = np.concatenate(templates, axis=0)[..., :-1]
#         # indices_start = tuple(len(t) for t in templates)[:-1]  # last index will go to end
#         # indices_cumulative = np.cumsum(indices_start)
#         # # filter out the invald indices
#         # mask = np.logical_and(np.all(coords >= 0, axis=1), np.all(coords <= tuple(s-1 for s in image.shape), axis=1))
#         # # count the correct indices for each template
#         # counts = tuple(np.count_nonzero(m) for m in np.split(mask, indices_cumulative))
#         # indices_cumulative_filtered = np.cumsum(counts)

#         # out = np.empty(indices_cumulative_filtered[-1], dtype=image.dtype)
#         # _index_floats_array_2d(coords[mask], image, out)
#         # a = np.split(out, indices_cumulative_filtered[:-1])
#     else:
#         raise ValueError(f"kind must be one of {kinds}.")

#     return out
