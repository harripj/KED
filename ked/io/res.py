from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Tuple, Union

import numpy as np
import orix
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from orix.quaternion.orientation import Orientation
from packaging import version


@dataclass
class ANG:
    """ANG class to read .ang data produced from ASTAR MapViewer."""

    file: Union[str, Path]
    data: pd.DataFrame
    header: dict
    shape: Tuple[int, int]  # in ij rather than xy
    pixel_size: Tuple[float, float]
    angle_keys: ClassVar[List] = ["psi1", "phi", "psi2"]

    def __post_init__(self):
        self.file = Path(self.file)

    def generate_image(self, key: str) -> NDArray:
        """
        Convenience function to reshape a signal.

        Parameters
        ----------
        key: str
            A key within data, eg. 'Index'.

        Returns
        -------
        out: ndarray
            Formatted data.
        """

        return self.data[key].values.reshape(self.shape)

    def plot(self, key: str, ax: plt.Axes = None) -> None:
        """
        Convenience plotting function.

        Parameters
        ----------
        key: str
            A key within data, eg. 'Index'.
        ax: plt.Axes or None
            Axes to plot on. If None a new figure is created.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.matshow(self.generate_image(key))

    @property
    def mask(self) -> NDArray:
        """Valid values mask. Invalid values have -ve index score."""
        return self.index >= 0

    @property
    def angles(self) -> NDArray:
        """Return the Euler angles in radians."""
        return self.data[self.angle_keys].values.reshape(
            *self.shape, len(self.angle_keys)
        )

    @property
    def orientations(self) -> Orientation:
        """Return the Euler angles as orix.quaternion.Orientation in
        'lab2crystal' reference frame, ie. Bunge convention."""
        # handle bug in orix
        if version.parse(orix.__version__) > version.parse("0.8.2"):
            direction = "lab2crystal"
        else:
            direction = "crystal2lab"
        # print(direction)
        return Orientation.from_euler(self.angles, direction=direction)

    @property
    def i(self):
        return self.generate_image("i")

    @property
    def j(self):
        return self.generate_image("j")

    @property
    def x(self):
        return self.generate_image("x")

    @property
    def y(self):
        return self.generate_image("y")

    @property
    def index(self):
        return self.generate_image("Index")

    @property
    def reliability(self):
        return self.generate_image("Reliability")

    @property
    def phase(self):
        return self.generate_image("Phase")

    @property
    def signal(self):
        return self.generate_image("Signal")

    @property
    def phases(self):
        return {k: v for k, v in self.header.items() if "phase" in k.lower()}

    def __repr__(self):
        return f"{self.__class__} {self.shape}: {self.file.stem}"

    @classmethod
    def from_file(cls, fname: Union[str, Path]):
        return cls.read_file(fname)

    @classmethod
    def read_file(cls, fname: Union[str, Path]) -> ANG:
        """

        Read .ang file produced by ASTAR MapViewer.

        Parameters
        ----------
        fname: str or Path
            Path to .ang file.

        Returns
        -------
        data: pd.DataFrame
            Map data.
        header: dict
            File header information.

        """
        strip_chars = "# \n"
        header = dict()

        with open(fname, "r") as f:
            same_phase = False

            for count, line in enumerate(f):
                line = line.strip()
                if not line.startswith("#"):
                    break

                if not line.strip(strip_chars):
                    # move on from empty lines at start of file
                    if not same_phase:
                        continue

                    header[phase[0]] = dict()

                    for k in phase[1:]:
                        if len(k) < 2:
                            k.append("")
                            header[phase[0]][k[0]] = k[1]
                        elif len(k) == 2:
                            try:
                                if "." in k[1]:
                                    header[phase[0]][k[0]] = float(k[1])
                                else:
                                    header[phase[0]][k[0]] = int(k[1])
                            except ValueError:
                                header[phase[0]][k[0]] = k[1]

                        else:
                            try:
                                header[phase[0]][k[0]] = tuple(float(i) for i in k[1:])
                            except ValueError:
                                header[phase[0]][k[0]] = tuple(k[1:])
                    same_phase = False
                    continue
                elif "Phase" in line:
                    phase = [line.strip(strip_chars)]
                    same_phase = True
                    continue
                else:
                    # only want these lines if not in a Phase block
                    if not same_phase:
                        header.setdefault("info", []).append(line.strip(strip_chars))

                # add phase information to list
                if same_phase:
                    phase.append(line.strip(strip_chars).split())

        df = pd.read_csv(
            fname,
            delimiter="\s+",
            skiprows=count,
            names=(
                "psi1",
                "phi",
                "psi2",
                "x",
                "y",
                "Index",
                "Reliability",
                "Phase",
                "Signal",
            ),
        )

        # due to rounding errors the values are not unique to 1dp even though they
        # should be -> round to 6dp
        x_unique = np.unique(np.round(np.ediff1d(np.sort(np.unique(df["x"]))), 6))
        y_unique = np.unique(np.round(np.ediff1d(np.sort(np.unique(df["y"]))), 6))

        # handle case where no scan along one dimension, ie. all x or y
        # => diff(x) or diff(y) is empty array
        if not x_unique.size:
            ps_x = 1
        elif x_unique.size > 1:
            print("x scan coordinates are not consistent with a single pixel size.")
            ps_x = 1
        else:
            ps_x = np.squeeze(x_unique)

        if not y_unique.size:
            ps_y = 1
        elif y_unique.size > 1:
            print("y scan coordinates are not consistent with a single pixel size.")
            ps_y = 1
        else:
            ps_y = np.squeeze(y_unique)
        # should be a unique number
        header["pixel_size"] = (float(ps_x), float(ps_y))

        # try to identify data shape
        try:
            ps_x, ps_y = header["pixel_size"]
            nx = (df["x"].max() - df["x"].min()) / ps_x
            ny = (df["y"].max() - df["y"].min()) / ps_y

            shape = (round(nx) + 1, round(ny) + 1)
            header["shape"] = shape

            df["i"] = (df["y"] / ps_y).round(0).astype(int)
            df["j"] = (df["x"] / ps_x).round(0).astype(int)
        except ValueError:
            # error raised due to dividing by list, when x or y have different deltas
            logging.warning("Data loaded but could not determine data shape.")

        # flip shape and pixel_size from xy to ij
        return cls(fname, df, header, header["shape"][::-1], header["pixel_size"][::-1])
