import os
from pathlib import Path

import numpy as np
import pandas as pd
from orix.quaternion.orientation import Orientation
from skimage import io as skio
from skimage import measure

from .components import filter_component_image


@pd.api.extensions.register_dataframe_accessor("components")
class Component:
    keys_angles = ["psi1", "phi", "psi2"]
    keys_positions = ["x", "y"]

    def __init__(self, data):
        assert all(
            i in data.columns for i in self.keys_angles
        ), f"{self.keys_angles} must be in the data."
        self._data = data

    @property
    def angles(self):
        return self._data[self.keys_angles].values

    @property
    def xy(self):
        return self._data[self.keys_positions].values

    def as_orientation(self):
        return Orientation.from_euler(np.deg2rad(self.angles), direction="lab2crystal")


def rename_component_images_MapViewer(files):
    """
    Rename component images files created through MapViewer.
    Image file names include '.res_comp' by default.
    This will be replaced by '_res_comp'.

    Parameters
    ----------
    files: array-like
        The files to rename.

    """
    for f in files:
        assert ".res_comp" in f, f".res_comp not found in file name: {f}"
        os.rename(str(f), str(f).replace(".res_comp", "_res_comp"))


def filter_components_files_MapViewer(
    files,
    fname=None,
    component_file=None,
    min_size=64,
    exclude_border=0,
    connectivity=1,
):
    """
    Use components images produced by MapViewer to separate and filter components.
    Return the center of mass (xy) and component index of each remaining component.

    Parameters
    ----------
    files: array-like of str or Path
        Component images files.
    fname: str or Path
        If provided then the data will be saved as .component (CSV) file.
    component_file: str or Path
        If provided then the Euler angles will be extracted from this file and appended
        to the .component file.
    min_size: int
        Smallest feature size in pixels to keep.
    exclude_border: int
        Components closer than this value (pixels) to border are ignored.
    connectivity: int
        The connectivity order. See skimage documentation.


    Returns
    -------
    coords: (N, 2) ndarray
        The xy coordinates of the remaining components.
    index: (N,) ndarray
        The component index of the remaining components.

    """
    files = [Path(f) for f in files]
    # sort by number
    files = sorted(files, key=lambda x: int(x.stem.split("_")[-1]))

    coords = []
    number = []

    for i, f in enumerate(files):
        im = skio.imread(f, as_gray=True)
        # apply filtering
        mask = filter_component_image(
            im > 0,
            min_size=min_size,
            exclude_border=exclude_border,
            connectivity=connectivity,
        )
        # label and get centroid from filtered mask
        labelled = measure.label(mask)
        rp = measure.regionprops(labelled)
        c = tuple(r.centroid for r in rp)
        # add to overall list
        coords.extend(c)
        # i + 1 to store ASTAR comp. index which starts from 1
        number.extend((i + 1,) * len(c))

    # convert coords ij to xy
    xy = np.array(coords)[..., ::-1]
    number = np.array(number)

    if fname is not None:
        _data = np.column_stack((xy, number))
        columns = ("x", "y", "index")
        if component_file is not None:
            _comp = load_components_MapViewer(Path(component_file))
            # -1 as component index starts from 1
            _data = np.column_stack((_data, _comp.components.angles[number - 1]))
            columns += tuple(Component.keys_angles)

        df = pd.DataFrame(_data, columns=columns).astype({"number": int})
        df.to_csv(Path(fname), sep="\t")

    return xy, number


def load_components_MapViewer(fname):
    """
    Read components list produced by MapViewer and extract information.

    Parameters
    ----------
    fname: str or Path
        Path to components.txt.

    Returns
    -------
    df: pd.DataFrame
        The formatted component information.
    """
    columns = (
        "Component",
        "psi1",
        "phi",
        "psi2",
        "Layer",
        "x",
        "y",
        "Index Component",
        "Index Total",
    )

    return pd.read_csv(fname, sep="\s+", skiprows=1, names=columns)
