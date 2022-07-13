from IPython.display import display
from ipywidgets import IntSlider, interactive
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def filter_matches_xy(matches, xy1, xy2, dx=None, dy=50, abs=True):
    """
    Filter matches between datasets by spatial displacement.

    Parameters
    ----------
    matches: tuple of ndarray of int
        The matching indices for each dataset.
    xy1, xy2: ndarray
        The xy coordinates of all components for each dataset.
    dx, dy: None or scalar
        The maximum allowed displacement in x and y.
    abs: bool
        If True use the absolute displacement as the filter criterion.

    Returns
    -------
    matches: tuple of ndarray of int
        The filtered matching indices.
    """

    delta = xy2[matches[1]] - xy1[matches[0]]
    # apply some spatial filtering

    # TODO: implement abs is False
    if abs is False:
        raise Warning("abs=False is not currently implemented.")

    # do filtering in each direction separately
    if dx is not None:
        val = delta[:, 0]
        if abs:
            val = np.abs(val)
        mask = val <= dx
        matches = tuple(idx[mask] for idx in matches)
    if dy is not None:
        val = delta[:, 1]
        if abs:
            val = np.abs(val)
        mask = val <= dy
        matches = tuple(idx[mask] for idx in matches)

    return matches


def sort_matches_multiple_couples(matches, comp1, comp2):
    """
    Sort coupling between component sets by minimum distance.
    In some cases many components are found on one component image.
    This function aims to sort the coupling using spatial information.

    Parameters
    ----------
    matches: tuple of ndarray of ints
        The matching indices. See output from linear_sum_assignment.
    comp1, comp2: pd.DataFrame
        The component information from the two datasets.
        Must contain 'index', 'x', and 'y' information.

    Returns
    -------
    matches: tuple of ndarray of ints
        The new matching indices considering sptial information.

    """
    _comp1 = comp1.iloc[matches[0]]
    _comp2 = comp2.iloc[matches[1]]

    # get the unique components from 1st dset, these are matched components
    unique = np.unique(_comp1["index"])

    # output from fn is new matches
    new_matches = np.empty_like(np.column_stack(matches))

    count = 0
    # only look at matched components
    for i, u in enumerate(unique):
        idx1 = np.nonzero(_comp1["index"].values == u)[0]

        if len(idx1) > 1:
            # check the positions of the matches
            xy1 = _comp1.iloc[idx1][["x", "y"]].values
            xy2 = _comp2.iloc[idx1][["x", "y"]].values

            # use overall minimum distance as matching metric
            dist = cdist(xy1, xy2)
            r, c = linear_sum_assignment(dist)
            count += np.count_nonzero(c != np.arange(len(c)))
        else:
            # do not change ordering
            r, c = np.arange(len(idx1)), np.arange(len(idx1))

        # update new matches
        new_matches[idx1, 0] = matches[0][idx1][r]
        new_matches[idx1, 1] = matches[1][idx1][c]

    print(f"{count} matches changed.")

    return tuple(
        new_matches.T
    )  # keep same format as from linear_sum_assignment (tuple)


def plot_matches(xy1, xy2, image1, image2, cmap="gray", color="k"):
    """
    Plot matches between datasets.

    Parameters
    ----------
    xy1, xy2: (N, 2) ndarray
        The xy posiitons of the components.
    image1, image2: ndarray
        An image of each sample, ed. VBF.
    cmap: str
        Colormap for image.
    color: str
        Colour for coupling lines.

    """
    fig, ax = plt.subplots()

    images = np.hstack((image1, image2))
    ax.matshow(images, cmap=cmap)

    x_offset = image1.shape[1]

    ax.plot(xy1[:, 0], xy1[:, 1], "r.")
    ax.plot(xy2[:, 0] + x_offset, xy2[:, 1], "b.")

    for _xy1, _xy2 in zip(xy1, xy2):
        ax.plot([_xy1[0], _xy2[0] + x_offset], [_xy1[1], _xy2[1]], color=color)


def plot_matches(coords1, coords2, im1, im2, matches, _interactive=False):
    """
    Plot coupled points and links (edges) on the same axes.

    Parameters
    ----------
    coords1, coords2: ndarray
        xy points for both datasets.
    im1, im2: ndarray
        Images for both datasets.
    matches: tuple of array-like
        The matching indices for coords1 and coords2.
    _interactive: bool
        If True then only one edge is shown at a time.
        A slider is presented to select the edge.

    """
    fig, ax = plt.subplots()
    ax.matshow(im1, extent=[0, im1.shape[1], im1.shape[0], 0])
    # plot im2 alongside
    x_offset = im1.shape[1]
    ax.matshow(im2, extent=[0 + x_offset, im1.shape[1] + x_offset, im2.shape[0], 0])

    # plot nodes
    ax.plot(coords1[:, 0], coords1[:, 1], "r.")
    ax.plot(coords2[:, 0] + x_offset, coords2[:, 1], "b.")

    # plot edges
    matches = np.asarray(matches)

    if _interactive:
        line = ax.plot([], [], "k", alpha=0.5)[0]

        def update(i):
            i, j = matches.T[i]
            line.set_xdata([coords1[i, 0], coords2[j, 0] + x_offset])
            line.set_ydata([coords1[i, 1], coords2[j, 1]])

        display(interactive(update, i=IntSlider(0, 0, len(matches.T) - 1)))
    else:
        for i, j in zip(*matches):
            ax.plot(
                [coords1[i, 0], coords2[j, 0] + x_offset],
                [coords1[i, 1], coords2[j, 1]],
                "k",
                alpha=0.5,
            )
