# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""atomsk as a reference: crystals made by atomsk --create. Test modules skip
where atomsk is not installed."""

import pathlib
import shutil
import subprocess
import tempfile

import mdapy

ATOMSK = shutil.which("atomsk")

# the names of the structures in atomsk
NAMES = {"l1_2": "L12"}


def atomsk_crystal(directory, structure, a, elements, *, c=None, orientation=None, repeat=(1, 1, 1)):  # fmt: skip
    """The crystal that atomsk --create makes, read back from a LAMMPS data
    file written in directory."""
    names = [elements] if isinstance(elements, str) else list(elements)
    command = [ATOMSK, "--create", NAMES.get(structure, structure), repr(a)]
    if c is not None:
        command.append(repr(c))
    command += names
    if orientation is not None:
        command.append("orient")
        command += ["[" + "_".join(str(int(i)) for i in d) + "]" for d in orientation]
    here = pathlib.Path(tempfile.mkdtemp(dir=directory))  # atomsk writes in it
    path = here / "crystal.lmp"
    command += ["-duplicate", *map(str, repeat), path.name, "-v", "0"]
    done = subprocess.run(
        command, cwd=here, capture_output=True, text=True, check=False
    )
    if not path.exists():
        raise RuntimeError(f"atomsk failed: {' '.join(command)}\n{done.stdout}")
    return mdapy.read(path)


def atomsk_apply(directory, system, *options, files=None):
    """The system after atomsk applies options to it, both through LAMMPS
    data files written in directory; files, {name: text}, are written there
    first (atomsk cuts long paths short)."""
    here = pathlib.Path(tempfile.mkdtemp(dir=directory))
    for name, text in (files or {}).items():
        (here / name).write_text(text)
    mdapy.write(here / "in.lmp", system)
    command = [ATOMSK, "in.lmp", *map(str, options), "out.lmp", "-v", "0"]
    done = subprocess.run(
        command, cwd=here, capture_output=True, text=True, check=False
    )
    if not (here / "out.lmp").exists():
        raise RuntimeError(f"atomsk failed: {' '.join(command)}\n{done.stdout}")
    return mdapy.read(here / "out.lmp")
