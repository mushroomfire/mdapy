Release Notes
===============

Mdapy 1.0.5a2 (April 30, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🐞 Bug Fixes
-------------

- ``System.__init__``: ``global_info`` no longer leaks across instances
  (mutable-default trap).
- ``System.cal_centro_symmetry_parameter``: respect the user's ``N``
  instead of a hardcoded 18 when sorting existing neighbors.
- ``System.cal_structure_entropy``: keep both ``entropy`` and
  ``entropy_ave`` columns when ``average_rc > 0``.
- ``System.cal_cluster_analysis``: invalid ``rc`` now raises
  ``TypeError`` (previously raised a bare string).
- ``System.write_data``: writing no longer mutates ``self.data`` —
  ``element_list`` is applied to a local copy.
- ``BuildSystem.from_ovito``: tolerate sources without a
  ``particle_type`` table (e.g. systems built from raw positions).
- ``cal_steinhardt_bond_orientation`` / ``cal_cluster_analysis`` /
  ``cal_chemical_species`` / ``cal_radial_distribution_function``:
  go through ``update_data`` so the DataFrame stays single-chunked
  and ``cal_radial_distribution_function`` no longer overwrites a
  user-provided ``type`` column.
- ``System.average_by_neighbor``: fix typo and removed unreachable
  branch.
- Fixed wrong ``write_mp`` docstring (was copied from ``replicate``).

🛠️ Other Improvements
----------------------

- ``System.box`` is now a property; assigning to it
  (``system.box = ...``) clears all neighbor / Voronoi / calculator
  caches automatically, matching ``update_box``'s contract.
- ``System.update_data``: misspelled ``reset_calcolator`` kwarg
  renamed to ``reset_calculator`` (old name kept as a deprecated
  alias with a ``DeprecationWarning``).
- Added ``System._get_compute_view()`` and collapsed 14 copies of
  the ``hasattr(self, "_enlarge_data")`` branch into a single helper.
- Added docstrings to ``write_xyz`` / ``write_poscar`` /
  ``write_data`` / ``write_dump``.
- New ``tests/test_system.py`` (47 tests) covers construction,
  mutators, calculator hook, neighbor invalidation, ASE/OVITO
  bridges, and regression tests for every fix above.
- ``Neighbor`` C++ kernel now bounds-checks every write. An
  over-tight ``max_neigh`` previously overran the buffer silently
  and caused memory corruption / segfaults; it now raises a
  ``ValueError`` reporting the required size. The fast path is
  unchanged when ``max_neigh`` is given correctly.
- ``Neighbor`` / ``NearestNeighbor`` constructors validate inputs
  (``rc`` / ``max_neigh`` / ``k`` ranges, required columns,
  non-empty data). ``knn.MAX_K`` is exposed as a module constant
  in place of the previous magic ``25``.

Mdapy 1.0.5a1 (April 28, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🏆 New Features
----------------

- Rewrote and optimized the k-nearest-neighbor algorithm; significantly
  faster than the previous implementation across all tested geometries.

🛠️ Other Improvements
----------------------

- Reworked the entire test suite to be self-contained: reference values
  for every algorithm are pre-computed and committed as small ``.npz``
  fixtures under ``tests/fixtures/``. Running the tests no longer
  requires OVITO, freud, ASE, pymatgen, pynep, or LAMMPS.
- Added a GitHub Actions workflow that runs the suite on every push and
  pull request (Windows + Linux + macOS, Python 3.11 / 3.12).

Mdapy 1.0.4 (April 24, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🏆 New Features
----------------

- Added support for rendering structures using the Tachyon method, accelerated by multi-core CPU and GPU (an order of magnitude faster).
- Added ``System.cal_build_bond()`` to generate bond index arrays directly from neighbor information, supporting a single cutoff as well as type-pair and element-pair cutoffs.
- ``Spline`` now supports three boundary conditions matches ``scipy.interpolate.CubicSpline``.

🐞 Bug Fixes
-------------

- Fixed the EAM potential file reader for LAMMPS ``eam/alloy`` format.
- Fixed EAM interpolation past the tabulated rho range. 

🛠️ Other Improvements
----------------------

- Added ``tests/test_spline.py`` comparing all three boundary
  conditions against ``scipy.interpolate.CubicSpline`` to floating-point
  precision.
- Refactor documentation, remove all warnings and errors.
- Add copyright for all codes.

Mdapy 1.0.3 (March 28, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🛠️ Other Improvements
----------------------

- Added explained_variance_ratio for PCA.
- For these analysis methods (CNA, PTM, and IDS), simulation boxes with a thickness below 15 Å are automatically treated as small boxes.
- The optimized elastic constant workflow can be validated using pymatgen.
- Improved the calculator setup behavior for the System.
- Fixed a bug in the minimize class.
- Updated parts of the documentation.


Mdapy 1.0.2 (March 4, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🛠️ Other Improvements
----------------------

- Reduced the size of the published source distribution (sdist).

- Added support for writing bec, stress, and virial fields in the comment line when using write_xyz.

- Improved LammpsPotential handling of general simulation boxes; it can now correctly compute stress, forces, and energy for such structures.

- Added newest version of NEPCPU for fixing a bug in qNEP.

Mdapy 1.0.1 (March 2, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🏆 New Features
----------------

- Added support for converting VASP OUTCAR files to XYZ format for GPUMD.
  This supports single-point calculations, structure minimization,
  and AIMD multi-frame trajectories.

- Added functionality to identify chemical species based on vdW-radius
  connectivity between atoms.

🛠️ Other Improvements
----------------------

- Fixed a bug in the HEA model builder.

- Optimized the cfg2xyz conversion tool.

- Added a neighbor-search benchmark to the documentation.

- Improved the energy minimization workflow.

- Optimized stress computation in ``LammpsPotential`` and added corresponding tests.

- Added support for saving BEC information in ``XYZTraj``.

- Added tests for qNEP.

- Removed warnings in ``test_void`` by introducing an explicit ``implode`` cast method.

🚀 Build System
----------------

- Binary wheels are now built separately for each Python version,
  instead of using the stable ABI (> Python 3.12). This change ensures
  compatibility with Python 3.13, where the stable ABI is not supported.


Mdapy 1.0.0 (January 3, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🚀 Summary
-----------
This is a milestone update for **mdapy**, featuring a near-complete rewrite of the core architecture. A primary driver for this transition was the limitations of our previous JIT dependency, **Taichi**; its development pace constrained our support for newer Python versions.

To ensure long-term sustainability, we have reconstructed mdapy by migrating computationally intensive kernels to C++ using the modern **nanobind** wrapper. As a result, mdapy now depends solely on **NumPy** and **Polars**, making it exceptionally lightweight and compatible with all modern Python environments.

**Key changes include:**

* **Engine Shift:** With the removal of Taichi, mdapy now focuses on high-performance CPU computation.
* **GUI & Tools:** Experimental Polyscope support has been removed to focus on core stability. However, a lightweight Jupyter-based GUI remains available as an optional dependency.
* **Modern Build System:** We have transitioned from ``setup.py`` to ``pyproject.toml``.
* **Reliability:** Extensive test suites have been added to ensure the correctness of all algorithms.

This is a brand-new foundation for the project, and we strongly recommend all users to upgrade.

🏆 New features
-----------------
* **qNEP Integration:** Support for evaluating energy, force, virial, charge, and BEC properties.
* **Structural Analysis:** Added Static Structure Factors and the Wigner-Seitz method for point defect detection.
* **Mechanical Properties:** Support for calculating elastic constants.
* **I/O Enhancements:** Added XYZ trajectory loading and introduced the **MP file format** (Parquet-based), providing high-speed I/O and efficient storage.
* **Minimization:** Improved FIRE2 method for energy minimization with cell optimization.
* **GPUMD Ecosystem:** A series of new features compatible with the GPUMD economy.

🛠️ Other improvements
----------------------
* **Compatibility:** Full support for Python >= 3.9.
* **Reliability:** Significantly expanded test case coverage.
* **Efficiency:** Optimized import overhead; ``import mdapy`` is now significantly faster.

⚠️ Limitations
---------------
* **Documentation:** The documentation is currently being updated and is not yet complete. We are actively working on this and welcome any community contributions or feedback.
