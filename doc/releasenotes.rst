Release Notes
===============

Mdapy 1.0.8a1 (June 12, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🐞 Bug Fixes
-------------

- Load XYZ files without cell information; Trajectory can now correctly read property data.
- kNN now can correctly handle general triclinic box.

Mdapy 1.0.7 (May 22, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🔧 API Changes
---------------

- :class:`mdapy.SQS`: simplified constructor; new
  ``is_sqs(tol=0.03, verbose=True)`` returns ``(passed, info)``.

- ``System.cal_build_bond()`` renamed to :meth:`System.create_bonds`.

- :func:`mdapy.build_crystal` Miller orientation now matches atomsk:
  cubic indices no longer need to obey the right-hand rule (left-handed
  frames give a mirror-image crystal), and hexagonal indices only need
  to be linearly independent — non-orthogonal frames produce a
  triclinic cell instead of raising.

🏆 New Features
----------------

- New :class:`mdapy.lammps_runner.LammpsRunner`: persistent LAMMPS
  driver for minimize / minimize_box / NVE / NVT / NPT on an mdapy
  :class:`System`.

- :class:`mdapy.LammpsPotential` accepts ``cmdargs`` and
  ``extra_commands`` for accelerator packages and pre-pair LAMMPS setup.

- New :class:`mdapy.QHAElastic`: temperature-dependent elastic constants
  via QHA (cubic + hexagonal), works with any mdapy calculator or
  external DFT. Requires ``phonopy``.

- New :meth:`System.delete_overlap`: remove atoms closer than a cutoff
  ``rc``, keeping the lowest-index atom of each overlapping cluster.


Mdapy 1.0.6 (May 6, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🏆 New Features
----------------

- New :class:`mdapy.SQS` for Special Quasirandom Structure generation.
- New :class:`mdapy.BondStiffness` for fitting longitudinal / transverse
  bond stiffnesses vs bond length.
- :class:`mdapy.RadialDistributionFunction` adds a streaming kernel for
  arbitrary cutoffs (no Verlet list materialised).
- :class:`mdapy.StructureFactor` rewritten in the Faber-Ziman convention
  of [Erhard2024]_, with X-ray / neutron / electron totals and the
  corresponding real-space PDFs.
- Partial outputs of ``RadialDistributionFunction`` /
  ``StructureFactor`` are now dicts keyed by element-pair tuples.
- New :func:`mdapy.unwrap_trajectory` / ``Trajectory.unwrap()``.
- Extended-XYZ I/O recognises GPUMD's ``unwrapped_position:R:3`` and
  aliases it to ``xu``/``yu``/``zu``.
- Unified thread-count control via the ``MDAPY_NUM_THREADS`` environment
  variable (set before ``import mdapy``); drives both OpenMP and polars.
  Defaults to ``os.cpu_count()``. Query via :func:`mdapy.get_num_threads`.
  The public ``workers`` / ``num_threads`` arguments on
  ``mdapy.compress_file`` and ``mdapy.render.TachyonRender`` are removed.
- New :meth:`System.cal_chill_plus` (CHILL+ algorithm) for identifying
  hexagonal ice, cubic ice, gas hydrate and interfacial phases in water
  simulations. 


🐞 Bug Fixes
-------------

- ``read_dump`` no longer fails (or silently overwrites) when a dump
  contains explicit ``x y z`` together with ``xs ys zs`` and/or
  ``xu yu zu``. Cartesian columns now take priority and the
  scaled/unwrapped variants stay under their original names; the
  fallback promotion only triggers when ``x y z`` is absent. Applies
  to both single-frame :class:`mdapy.System` and multi-frame
  :class:`mdapy.Trajectory` loaders.

Mdapy 1.0.5 (May 1, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🏆 New Features
----------------

- ``build_crystal`` extended to match ``atomsk --create`` byte-for-byte:
  added ``sc``, ``rocksalt``, ``cscl``, ``zincblende``, ``fluorite``,
  ``L1_2``, ``perovskite``, ``wurtzite``, ``graphite`` and
  ``lonsdaleite``; multi-species via tuple ``name``; Miller orientation
  for every cubic and Miller-Bravais ``[hkil]`` for every hexagonal
  structure. ``hcp`` and ``graphene`` now use atomsk's 2-atom hex
  primitive (atom counts halve compared with the legacy supercell).
- New ``mdapy.orthogonal_cell(system, find_minimal=False)`` — atomsk's
  ``-orthogonal-cell`` option.
- New unified ``mdapy.Trajectory`` class for multi-frame XYZ + LAMMPS
  dump (read + write). Supports fancy indexing with integer arrays
  and boolean masks (e.g. ``traj[traj.get_atoms_count() > 100]``),
  optional ``fast_mode=True`` for vectorised XYZ reading (~5–7×
  speedup on regular files), an in-place progress bar
  (``verbose=True``, default on), and a ``vacuum=`` option on
  ``save("out.xyz", vacuum=200.0)`` that pads every non-periodic axis
  by ``vacuum`` Å and centres the cluster — useful for auto-boxing
  classical-XYZ training-set frames so downstream MD code sees a
  well-defined supercell.


🐞 Bug Fixes
-------------

- ``Neighbor`` C++ kernel now bounds-checks every write: an over-tight
  ``max_neigh`` raises ``ValueError`` instead of silently corrupting
  memory.
- ``System``: mutable-default ``global_info`` no longer leaks across
  instances; ``cal_centro_symmetry_parameter`` honours the user's
  ``N``; ``cal_structure_entropy`` keeps both columns when
  ``average_rc>0``; ``cal_cluster_analysis`` raises ``TypeError`` on
  bad ``rc``; ``write_data`` no longer mutates ``self.data``;
  ``cal_radial_distribution_function`` no longer overwrites the user's
  ``type`` column.
- ``BuildSystem.from_ovito`` tolerates sources without a
  ``particle_type`` table.
- IO writers: ``write_xyz`` Properties string assembly; ``write_dump``
  preserves the ``element`` column; cross-device temp files on Windows.

🛠️ Other Improvements
----------------------

- Rewrote and optimized the k-nearest-neighbor algorithm; significantly
  faster than the previous implementation across all tested geometries.
- API: ``c_over_a`` removed from ``build_crystal`` / ``build_hea`` —
  pass ``c`` directly. ``System.box`` is now a property whose setter
  invalidates neighbor / Voronoi / calculator caches automatically.
  ``System.update_data``: misspelled ``reset_calcolator`` renamed to
  ``reset_calculator`` (old name deprecated).
- Hardened IO: rewrote LAMMPS data / dump / XYZ / POSCAR / .mp readers
  to be format-robust (multi-space, CRLF, abc-origin triclinic,
  Masses element comments, image flags, etc.) and self-consistent
  on round-trip. ``Neighbor`` / ``NearestNeighbor`` constructors
  validate inputs.
- Internal cleanup: 14 ``_enlarge_data`` branches collapsed into one
  helper; ``cal_*`` methods uniformly route through ``update_data`` so
  the DataFrame stays single-chunked; legacy ``_get_basispos_and_box_cubic``
  removed.
- Tests: +85 cases (``test_system.py``, ``test_build_crystal.py``,
  ``test_orthogonal_cell.py``, ``test_io_*``); IO references match
  atomsk byte-for-byte. CI is fully self-contained.
- Trajectory IO speedups: per-frame XYZ parser now picks ``pl.read_csv``
  (uniform single-space) or numpy split + per-column ``pl.Series``
  (multi-space); the previous Python dict-of-lists path was dropped.
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
- Added ``System.create_bonds()`` to generate bond index arrays directly from neighbor information, supporting a single cutoff as well as type-pair and element-pair cutoffs.
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
