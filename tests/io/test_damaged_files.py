# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Damaged files of every format give an error of MDAPY, never another one.

Each sample file is damaged at random, from a fixed seed, as files are: cut short,
lines lost or doubled, bytes changed, numbers replaced by others. Reading the result
must give a system or an MdapyError (FileFormatError for most), for one frame and for
a whole trajectory.
"""

import random

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.errors import MdapyError

TOKENS = [
    b"nan", b"inf", b"-1", b"0", b"1e308", b"99999999999999999999", b"x", b"", b"T",
    b"1.5", b"-0", b"2147483648", b"-9223372036854775809",
]  # fmt: skip
INSERTS = [b"\n", b"\r\n", b"\t", b"  ", b"#", b'"', b"=", b"\x00"]


def damaged(data: bytes, r: random.Random) -> bytes:
    """data with one to eight random kinds of damage."""
    b = bytearray(data)
    for _ in range(r.choice([1, 1, 1, 2, 3, 8])):
        if not b:
            break
        i = r.randrange(len(b))
        lines = bytes(b).split(b"\n")
        kind = r.randrange(9)
        if kind == 0:
            b[i] = r.randrange(256)
        elif kind == 1:
            del b[i:]
        elif kind == 2:
            b[i:i] = bytes(r.randrange(256) for _ in range(r.randrange(1, 8)))
        elif kind == 3 and len(lines) > 1:
            del lines[r.randrange(len(lines))]
            b = bytearray(b"\n".join(lines))
        elif kind == 4 and len(lines) > 1:
            k = r.randrange(len(lines))
            lines.insert(k, lines[k])
            b = bytearray(b"\n".join(lines))
        elif kind == 5:
            tokens = bytes(b).split(b" ")
            tokens[r.randrange(len(tokens))] = r.choice(TOKENS)
            b = bytearray(b" ".join(tokens))
        elif kind == 6:
            del b[i : i + r.randrange(1, 40)]
        elif kind == 7:
            j = r.randrange(len(b))
            b[i], b[j] = b[j], b[i]
        else:
            b[i:i] = r.choice(INSERTS)
    return bytes(b)


def samples(tmp_path) -> dict[str, tuple[str, str, bytes]]:
    rng = np.random.default_rng(1)
    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(2, 2, 2))
    system = system.with_columns(
        velocities=rng.normal(size=(len(system), 3)),
        forces=rng.normal(size=(len(system), 3)),
        charge=rng.normal(size=len(system)),
    ).with_info(energy=-12.5)
    files = {}
    for fmt, name, frames in [
        ("extxyz", "a.xyz", [system, system]),
        ("lammps-data", "a.data", system),
        ("lammps-dump", "a.dump", [system, system]),
        ("poscar", "POSCAR", system),
        ("mdapy", "a.mdapy", [system, system]),
    ]:
        mdapy.io.write(tmp_path / name, frames, fmt)
        files[fmt] = (fmt, name, (tmp_path / name).read_bytes())
    files["plain xyz"] = (
        "extxyz",
        "b.xyz",
        b"3\nMMM12 made of C atoms; MMM12 logo\nC 0 0 0\nH 1.0 0 0\nH 0 1.0 0\n",
    )
    files["cif"] = (
        "cif",
        "a.cif",
        b"data_x\n_cell_length_a 3.6\n_cell_length_b 3.6\n_cell_length_c 3.6\n"
        b"_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        b"_symmetry_space_group_name_H-M 'F m -3 m'\nloop_\n_atom_site_label\n"
        b"_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
        b"_atom_site_fract_z\nCu1 Cu 0 0 0\n",
    )
    files["pdb"] = (
        "pdb",
        "a.pdb",
        b"CRYST1    3.600    3.600    3.600  90.00  90.00  90.00 P 1           1\n"
        b"ATOM      1  CU  CU  A   1       0.000   0.000   0.000  1.00  0.00          CU\n"
        b"ATOM      2  CU  CU  A   1       1.800   1.800   0.000  1.00  0.00          CU\n"
        b"END\n",
    )
    return files


@pytest.mark.parametrize(
    "sample",
    [
        "extxyz",
        "plain xyz",
        "lammps-data",
        "lammps-dump",
        "poscar",
        "mdapy",
        "cif",
        "pdb",
    ],
)
def test_damaged_files_give_errors_of_mdapy(tmp_path, sample):
    fmt, name, data = samples(tmp_path)[sample]
    r = random.Random(f"{sample} 20261106")
    path = tmp_path / ("damaged-" + name)
    for _ in range(120):
        text = damaged(data, r)
        path.write_bytes(text)
        try:
            mdapy.io.read(path, fmt)
        except MdapyError:
            pass
        except Exception as e:
            pytest.fail(f"{type(e).__name__}: {e} reading {text[:200]!r}")
        try:
            with mdapy.io.read_trajectory(path, fmt) as trajectory:
                for _frame in trajectory:
                    pass
        except MdapyError:
            pass
        except Exception as e:
            pytest.fail(f"{type(e).__name__}: {e} reading frames of {text[:200]!r}")
