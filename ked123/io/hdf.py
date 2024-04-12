import logging
import math

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_frame_key(n):
    """
    Return the dictionary key for frame number in hdf5 file from TVIPSconverter.

    Parameters
    ----------
    n: int
        Frame number.

    Returns
    -------
    key: str
        Frame key for 'ImageStream' group.
    """
    return f"Frame_{str(n).zfill(6)}"


def read_scan_settings(fname):
    """
    Read scan parameters from TVIPSconverter HDF5 data.
    Data is stored as attrs of h5['Scan'] group.

    Parameters
    ----------
    fname: str or Path
        Path to hdf5.

    Returns
    -------
    parameters: dict
        The scan settings stored when calculating the VBF image.
    """
    with h5py.File(fname, "r") as h5:
        group = h5["Scan"]

        out = dict(**group.attrs)

    return out


def read_attrs_hdf5(fname, numbers):
    """
    Get the attributes from an individual frame or set of frames from a hdf5 file produced by TVIPSconverter.

    Parameters
    ----------
    fname: str or Path
        Path to .hdf5 file.
    numbers: int, tuple or list of ints or None
        Frame numbers to return.
        If None then all frames will be read.

    Returns
    -------
    frames: ndarray or iterable
        Returned frames stacked along the first dimension.
    """

    if numbers is None:
        # frames are defined as zfill(6) => max n is < 1e7
        numbers = range(int(1e7))
        BREAK_ON_KEYERROR = True
    elif isinstance(numbers, (int, np.integer)):
        numbers = (numbers,)
        BREAK_ON_KEYERROR = False
    else:
        BREAK_ON_KEYERROR = False

    out = []

    with h5py.File(fname, "r") as h5:
        group = h5["ImageStream"]

        # load the frames
        for n in tqdm(numbers, desc="Progress"):
            key = get_frame_key(n)
            try:
                attrs = dict(**group[key].attrs)
            except KeyError:
                if BREAK_ON_KEYERROR:
                    break
                else:
                    logging.warning(f"{key} does not exist.")
                    continue

            out.append(attrs)

    if len(numbers) == 1:
        out = out.pop()

    return out


def read_frame_hdf5(fname, numbers, attrs=False):
    """
    Get an individual frame or set of frames from a hdf5 file produced by TVIPSconverter.

    Parameters
    ----------
    fname: str or Path
        Path to .hdf5 file.
    numbers: int, tuple or list of ints or None
        Frame numbers to return.
        If None then all frames will be read.
    attrs: bool
        If True then the frame attributes are also returned.

    Returns
    -------
    frames: ndarray or iterable
        Returned frames stacked along the first dimension.
    """

    if numbers is None:
        # frames are defined as zfill(6) => max n is < 1e7
        numbers = range(int(1e7))
        BREAK_ON_KEYERROR = True
    elif isinstance(numbers, (int, np.integer)):
        numbers = (numbers,)
        BREAK_ON_KEYERROR = False
    else:
        BREAK_ON_KEYERROR = False

    out = []
    out_attrs = []

    with h5py.File(fname, "r") as h5:
        group = h5["ImageStream"]

        # load the frames
        for n in tqdm(numbers, desc="Progress"):
            key = get_frame_key(n)
            try:
                frame = np.array(group[key])
            except KeyError:
                if BREAK_ON_KEYERROR:
                    break
                else:
                    logging.warning(f"{key} does not exist.")
                    continue

            out.append(frame)
            if attrs:
                out_attrs.append(dict(**group[key].attrs))

    if len(numbers) == 1:
        out = out.pop()
        if attrs:
            out_attrs = out_attrs.pop()

    return (out, out_attrs) if attrs else out


def calculate_frame_number(ij, scan_shape, crop_start=(0, 0), frame_offset=0):
    """
    Calculate the frame number within the hdf5 file produced by TVIPSconverter.

    Parameters
    ----------
    ij: tuple or (N, 2) integer array
        Tuple of length len(scan_shape), ie. one coordinate for each dimension.
        Otherwise array where second axis has length len(scan_shape), ie. columns of row (i) and column (j) (etc.) of indices in question.
    scan_shape: tuple
        Overall shape of experimental scan (from TVIPS).
    crop_start: tuple
        The ij of the initial crop position if querying the location of a pixel within cropped data of the original scan.
    frame_offset: int
        Initial scan frame offset within experimental data (from TVIPS).

    Returns
    -------
    indices: int
        Frame index within experimental hdf5 file.
    """

    return (
        np.ravel_multi_index(np.transpose(np.asarray(ij) + crop_start), scan_shape)
        + frame_offset
    )


def read_scan_shape(fname):
    """
    Read the calculated scan shape and frame offset from TVIPS .hdf5 file.

    Parameters
    ----------
    fname: str or Path
        Path to .hdf5 file.

    Returns
    -------
    shape: tuple
        Scan shape (i, j), ie. (y, x).
    offset: int
        Scan frame offset.
    """
    settings = read_scan_settings(fname)

    shape = (settings["scan_dim_y"], settings["scan_dim_x"])

    return shape, settings["start_frame"]


def read_vbf(fname):
    """
    Read the calculated Virtual Bright Field reconstruction from TVIPS .hdf5 file.

    Parameters
    ----------
    fname: str of Path
        Path to .hdf5 file.

    Returns
    -------
    vbf: ndarray
        VBF reconstruction.
    """

    shape, offset = read_scan_shape(fname)

    vbf = read_calculated_parameter(fname, "vbf_intensities")

    out = np.reshape(vbf[offset : offset + math.prod(shape)], shape)

    return out


def read_calculated_parameter(fname, key=None):
    """
    Read calculated parameter from TVIPSconverter HDF5 data.
    Data is stored at h5['Scan'].

    Parameters
    ----------
    fname: str or Path
        Path to hdf5.
    key: str or None
        Key to read.
        If None then a tuple of available keys is returned.

    Returns
    -------
    data: The data associated with key, or the list of available keys if key was None.
    """

    with h5py.File(fname, "r") as h5:
        group = h5["Scan"]

        if key is None:
            out = tuple(group.keys())
        else:
            out = np.array(group[key])

    return out
