# Fixture generators

This directory holds the **one-time** scripts that build the reference
data under `tests/fixtures/`. The generated `.npz` files are committed
to the repo, so the test suite itself never has to import OVITO,
freud, sklearn, ASE, pymatgen, pynep, or LAMMPS.

`pytest` automatically skips this directory (see `conftest.py`) so the
scripts here never run as tests.

## Layout

```
configs.py                          shared crystal/random configurations
                                    + per-test overrides (skip lists, k values, cutoffs)

generate_structure_analysis.py      OVITO + freud → per-atom labels for
                                    CSP / CNA / AJA / PTM / IDS / CNP / Q_l / Voronoi
                                    Output: tests/fixtures/structure_analysis/<config>.npz

generate_misc.py                    OVITO + freud → standalone reference data for
                                    atomic_strain, wigner_seitz, RDF, ADF, bond,
                                    structure factor, MSD, structure entropy,
                                    average_neighbor, FCC planar faults
                                    Output: tests/fixtures/misc/<algo>.npz

generate_advanced.py                sklearn / ASE / pymatgen / pynep / LAMMPS →
                                    PCA, build_crystal, elastic constant,
                                    minimization, NEP potential
                                    Output: tests/fixtures/advanced/<algo>.npz
```

## When to refresh

Re-run the matching generator whenever:

* The mdapy algorithm itself is modified (so a fixture mismatch is the
  intended signal).
* Reference parameters (cutoffs, num_neighbors, modes) change.
* A new configuration is added to `configs.py`.
* A new algorithm is added — copy an existing `_xxx(name, system)`
  function inside the appropriate `generate_*.py` and add the test file.

After refreshing, `git diff tests/fixtures/` shows what changed; sanity-
check it before committing.

## How to run

The scripts require the corresponding upstream library installed in the
current Python env:

```bash
# Requires: ovito, freud
python tests/_generate_fixtures/generate_structure_analysis.py
python tests/_generate_fixtures/generate_misc.py

# Requires: sklearn, ase, pymatgen, pynep, lammps (with NEP plugin)
python tests/_generate_fixtures/generate_advanced.py
```

`generate_advanced.py` tolerates partial environments — if one
generator fails (e.g. LAMMPS missing), it logs the error and continues
with the others. Tests for fixtures that weren't regenerated still pass
against the previously committed values.

## What goes in a fixture

* Inputs needed to reconstruct the test scenario (parameters, seeds,
  filenames pointing at `tests/input_files/`).
* Reference outputs from the upstream library (per-atom arrays,
  histograms, scalar tensors).

What does **not** go in a fixture:

* Bulk trajectories or large random arrays — store the seed and let the
  test reconstruct them.
* Anything that mdapy itself would compute deterministically from the
  inputs already saved.

Keep individual fixtures small (≪ 100 KB) so the repo stays git-friendly.
