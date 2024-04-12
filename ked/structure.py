import re
from pathlib import Path
from typing import Union

import numpy as np
from ase import Atom as aseAtom
from ase import Atoms as aseAtoms
from ase.data import chemical_symbols
from diffpy.structure import Atom as diffpyAtom
from diffpy.structure import Lattice, Structure
from numpy.typing import NDArray
from orix.crystal_map import Phase


def parse_structure(
    structure: Union[aseAtoms, Phase, Structure, Path, str]
) -> Structure:
    """Parse a structure input."""
    # sort out phase, use conventions defined in orix
    if isinstance(structure, (Path, str)):
        phase = Phase.from_cif(str(structure))
    elif isinstance(structure, Structure):
        # performs axes realignment
        phase = Phase(structure=structure)
    elif isinstance(structure, aseAtoms):
        phase = Phase(
            structure=Structure(
                atoms=[
                    diffpyAtom(atype=atom.symbol, xyz=atom.scaled_position)
                    for atom in structure
                ],
                lattice=Lattice(base=structure.get_cell().array),
            )
        )
    elif isinstance(structure, Phase):
        phase = structure
    else:
        raise TypeError(f"Structure with type: {type(structure)} not supported.")

    return phase.structure


def get_positions(structure: Union[aseAtoms, Structure, Phase]) -> NDArray:
    """Return atomic cartesian coordinates in Angstroms."""
    if isinstance(structure, Phase):
        structure = structure.structure
    if isinstance(structure, aseAtoms):
        positions = structure.get_positions()
    elif isinstance(structure, Structure):
        positions = structure.xyz_cartn
    else:
        raise TypeError(f"{type(structure)} not supported.")
    return positions


def get_scaled_positions(structure: Union[aseAtoms, Structure, Phase]) -> NDArray:
    """Return atomic coordinates as fraction of unit cell."""
    if isinstance(structure, Phase):
        structure = structure.structure
    if isinstance(structure, aseAtoms):
        scaled_positions = structure.get_scaled_positions()
    elif isinstance(structure, Structure):
        scaled_positions = structure.xyz
    else:
        raise TypeError(f"{type(structure)} not supported.")

    return scaled_positions


def get_unit_vectors(structure: Union[aseAtoms, Structure, Phase]) -> NDArray:
    """Return lattice unit vectors as row vectors in Angstroms."""
    if isinstance(structure, Phase):
        structure = structure.structure
    if isinstance(structure, aseAtoms):
        cell = structure.get_cell().array
    elif isinstance(structure, Structure):
        cell = structure.lattice.base
    else:
        raise TypeError(f"{type(structure)} not supported.")

    return cell


def get_unit_cell_volume(structure: Union[aseAtoms, Structure, Phase]) -> float:
    """Return the unit cell volume in cubic Angstroms."""
    if isinstance(structure, Phase):
        structure = structure.structure
    if isinstance(structure, aseAtoms):
        volume = structure.get_cell().volume
    elif isinstance(structure, Structure):
        volume = structure.lattice.volume
    else:
        raise TypeError(f"{type(structure)} not supported.")

    return volume


def remove_charge_from_element_string(element: str) -> str:
    """Remove charge from element string, eg. 0+ from Ni0+."""
    match = re.search("[0-9][+-]", element)
    if match is None:
        out = element
    else:
        to_remove = element[match.start() : match.end()]
        out = element.replace(to_remove, "")
    return out


def get_element_name(
    element: Union[aseAtoms, diffpyAtom, str, int], remove_charge: bool = True
) -> str:
    """
    Convenience function to return the element name as a string.

    Parameters
    ----------
    element: str, ase.Atom, diffpy.structure.Atom, int
        Element identifier may be an Atom, string, or atomic number.
    remove_charge: bool
        If True then charge, eg. '1+', is removed from the element
        string.

    Returns
    -------
    name: str
        The chemical abbreviation for the element.
    """
    if isinstance(element, aseAtom):
        element = chemical_symbols[element.number]
    elif isinstance(element, diffpyAtom):
        element = element.element
    elif isinstance(element, (int, np.integer)):
        element = chemical_symbols[element]
    elif isinstance(element, str):
        pass
    else:
        raise TypeError(f"Element with type {type(element)} is not supported.")

    if not isinstance(element, str):
        raise TypeError(
            f"Element: {element} is not of the correct type. "
            + "Element should be either int, str or ase.Atom or diffpy.structure.Atom."
        )

    if remove_charge:
        element = remove_charge_from_element_string(element)

    return element
