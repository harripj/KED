from pathlib import Path
from typing import Callable, Tuple

from ase import io as aseio
from diffpy.structure import Structure, loadStructure
import numpy as np
from orix.crystal_map import Phase
import pytest

from ked.structure import (
    get_positions,
    get_scaled_positions,
    get_unit_cell_volume,
    get_unit_vectors,
    parse_structure,
)


def get_parsed_structures(
    file: Path,
) -> Tuple[Structure, Structure, Structure, Structure]:
    file = str(file)
    a = aseio.read(file)
    s = loadStructure(file)
    p = Phase.from_cif(file)
    return (
        parse_structure(s),
        parse_structure(a),
        parse_structure(p),
        parse_structure(file),
    )


def test_parse_structure(cif_files):
    for file in cif_files:
        s, a, p, f = get_parsed_structures(file)
        assert all(isinstance(i, Structure) for i in (s, a, p, f))


@pytest.mark.parametrize(
    "fn", [get_positions, get_scaled_positions, get_unit_vectors, get_unit_cell_volume]
)
def test_get_functions(fn: Callable, cif_files):
    for file in cif_files:
        s, a, p, f = get_parsed_structures(file)
        ps = fn(s)
        assert all(np.allclose(ps, fn(x)) for x in (a, p, f))
