from pathlib import Path

from ase import Atoms
from ase import io as aseio
from diffpy.structure import Structure, loadStructure
import numpy as np
import py
import pytest

import KED


@pytest.fixture
def cif_Fe_BCC():
    file = Path(KED.__file__).parent.joinpath("data", "testing", "Fe alpha.cif")
    assert file.exists(), "File not found."
    return file.relative_to(Path.cwd())


@pytest.fixture
def cif_Mg_hexagonal():
    file = Path(KED.__file__).parent.joinpath("data", "testing", "Mg.cif")
    assert file.exists(), "File not found."
    return file.relative_to(Path.cwd())


@pytest.fixture
def cif_ReS2_triclinic():
    file = Path(KED.__file__).parent.joinpath("data", "testing", "ReS2.cif")
    assert file.exists(), "File not found."
    return file.relative_to(Path.cwd())


# TODO: ase and diffpy need comparing for non-cubic crystals
@pytest.fixture(params=["cif_Fe_BCC", "cif_Mg_hexagonal"])
def cif_files(request):
    return request.getfixturevalue(request.param)


def atoms(cif):
    return aseio.read(cif)


def structure(cif):
    return loadStructure(str(cif))


def test_load_cif(cif_files):
    a = atoms(cif_files)
    s = structure(cif_files)
    assert isinstance(a, Atoms)
    assert isinstance(s, Structure)
    assert len(a) == len(s)


def test_unit_cell(cif_files):
    a = atoms(cif_files)
    ca = a.get_cell()
    s = structure(cif_files)
    cs = s.lattice
    assert np.allclose(ca.array, cs.base)
    assert np.allclose(ca.volume, cs.volume)


def test_atomic_positions(cif_files):
    a = atoms(cif_files)
    s = structure(cif_files)
    assert np.allclose(a.get_positions(), s.xyz_cartn)
