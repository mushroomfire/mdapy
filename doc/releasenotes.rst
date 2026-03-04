Release Notes
===============

Mdapy 1.0.2 (March 4, 2026)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🛠️ Other Improvements
----------------------

- Reduced the size of the published source distribution (sdist).

- Added support for writing bec, stress, and virial fields in the comment line when using write_xyz.

- Improved LammpsPotential handling of general simulation boxes; it can now correctly compute stress, forces, and energy for such structures.

- Added newest version for fixing a bug in qNEP.

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
