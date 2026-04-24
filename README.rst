.. raw:: html

   <div align="center">
     <img src="https://raw.githubusercontent.com/mushroomfire/mdapy/master/doc/_static/mdapy-logo.svg" alt="mdapy logo" width="760">
   </div>

|

.. image:: https://img.shields.io/pypi/v/mdapy.svg
   :target: https://pypi.org/project/mdapy/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/mdapy.svg
   :target: https://pypi.org/project/mdapy/
   :alt: Python versions

.. image:: https://img.shields.io/pypi/dm/mdapy.svg
   :target: https://pypi.org/project/mdapy/
   :alt: PyPI downloads

.. image:: https://img.shields.io/github/license/mushroomfire/mdapy.svg
   :target: https://github.com/mushroomfire/mdapy/blob/master/LICENSE
   :alt: License

.. image:: https://readthedocs.org/projects/mdapy/badge/?version=latest
   :target: https://mdapy.readthedocs.io/en/latest/
   :alt: Documentation

.. image:: https://img.shields.io/github/stars/mushroomfire/mdapy.svg?style=social
   :target: https://github.com/mushroomfire/mdapy
   :alt: GitHub stars

|

**mdapy** (Molecular Dynamics Analysis in Python) is a fast, full-featured
Python library for analyzing Molecular Dynamics (MD) simulation data — from
structural characterization and machine-learning potential workflows to built-in
ray-tracing visualization, all in a clean Pythonic API.

.. code-block:: bash

   pip install mdapy

|

----

Why mdapy?
----------

.. list-table::
   :widths: 5 25 70
   :header-rows: 0

   * - ⚡
     - **Blazing Fast**
     - C++ core (via Nanobind) with full OpenMP parallelism. Analyses that take minutes
       elsewhere finish in seconds.

   * - 🐍
     - **Pythonic by Design**
     - One-liner analyses. All data lives in a
       `Polars <https://pola.rs>`_ DataFrame and interops natively with NumPy.

   * - 🪶
     - **Minimal Dependencies**
     - The core package depends only on NumPy and Polars, keeping installation
       lightweight and easy to reproduce across environments.

   * - 🌍
     - **Truly Cross-Platform**
     - Pre-built wheels for Windows, Linux, and macOS (including Apple Silicon).
       No compiler needed for standard installs.

   * - 🎨
     - **Built-in Ray-Tracing**
     - Tachyon-powered CPU *and* GPU (NVIDIA OptiX) renderer baked right in —
       no third-party renderer required.

   * - 🤝
     - **Ecosystem Friendly**
     - First-class interop with OVITO, ASE, freud, phonopy, LAMMPS and GPUMD.

   * - 📖
     - **Thoroughly Documented**
     - Full API reference, tutorials, and Jupyter notebooks on
       `ReadTheDocs <https://mdapy.readthedocs.io>`_.

   * - 🔄
     - **Actively Maintained**
     - Regular releases, responsive issue tracker, and a growing feature set.

----

Feature Overview
----------------

Neighbor Search
~~~~~~~~~~~~~~~

+---------------------------+---------------------------------------------+
| Fixed-radius cutoff       | Efficient cell-list algorithm               |
+---------------------------+---------------------------------------------+
| k-Nearest neighbors       | Exact kNN via aabb tree                     |
+---------------------------+---------------------------------------------+
| Voronoi neighbors         | Topology-based, powered by Voro++           |
+---------------------------+---------------------------------------------+

Structural Analysis
~~~~~~~~~~~~~~~~~~~

- **Structure identification** — Polyhedral Template Matching (PTM),
  Common Neighbor Analysis (CNA), Ackland-Jones Analysis,
  Common Neighbor Parameter, Identify Diamond Structure
- **Defect characterization** — Centro-Symmetry Parameter (CSP),
  Identify FCC Planar Faults (ISF / TB / ESF), Atomic Strain,
  Wigner-Seitz defect analysis
- **Order parameters** — Steinhardt Bond Orientation (q₂…q₁₂, averaged),
  Structure Entropy, Warren-Cowley SRO parameter
- **Distributions** — Radial Distribution Function, Structure Factor
  (total + partial + X-ray weighted), Angular Distribution Function,
  Bond Analysis

Model Building
~~~~~~~~~~~~~~

- Single-crystal lattices (FCC, BCC, HCP, diamond, and more)
- **Large-scale polycrystalline** structures via Voronoi tessellation
- High-entropy alloy (HEA) configurations

Machine-Learning Potential Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **NEP / qNEP** potential evaluation (energy, forces, stress, virials)
- ASE-compatible ``NEP4ASE`` calculator
- Elastic constant tensor calculation
- Equation-of-state (EOS) fitting
- Stacking-fault energy curves
- Phonon dispersion via phonopy
- Dataset extraction, PCA, FPS, display Train/Thermo results, etc.

Visualization
~~~~~~~~~~~~~

- **TachyonRender** — CPU or NVIDIA OptiX GPU ray-tracing, renders
  any ``System`` to a PNG/JPEG with one method call
- Per-element colors and radii, transparent background, anti-aliasing,
  ambient occlusion, shadows — all configurable

Utilities
~~~~~~~~~

- EAM potential generation and averaging
- Mean Squared Displacement (FFT-accelerated or direct)
- Lindemann parameter, Void analysis, Cluster analysis
- Spatial binning (multi-dimensional)
- Atomic temperature averaging
- Parallel ``.gz`` compression
- Multi-frame XYZ trajectory reader / splitter

----

Installation
------------

**One command (recommended):**

.. code-block:: bash

   pip install mdapy

**With optional extras:**

.. code-block:: bash

   pip install mdapy[plot]      # + matplotlib
   pip install mdapy[k3d]       # + k3d 3-D widget
   pip install mdapy[all]       # + matplotlib, k3d, pyfftw, phonopy

**From the PyPI source distribution** (requires C++17 compiler + OpenMP):

If no pre-built wheel fits your platform, build from the sdist we also
publish to PyPI. It is the smallest way to build from source — no git
history, no docs, no test inputs — and pip handles the download:

.. code-block:: bash

   pip install --no-binary mdapy mdapy

The ``--no-binary mdapy`` flag skips the wheel for mdapy only;
dependencies (``numpy``, ``polars``) still install as wheels.

**From a git clone** (recommended for development):

.. code-block:: bash

   git clone https://github.com/mushroomfire/mdapy.git
   cd mdapy
   pip install .                  # regular install
   # or:
   pip install -e .               # editable, for modifying the source

Tested compilers: MSVC (Windows 10), GCC (Ubuntu), Clang (macOS M1).

**Verify:**

.. code-block:: bash

   python -c "import mdapy as mp; print('mdapy', mp.__version__, '— ready!')"

Interoperability with PyTorch / OVITO / freud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mdapy, PyTorch, OVITO, freud and scikit-learn all use OpenMP for internal
parallelism. Because each project ships its own pre-built binaries, a
single Python process can end up with more than one copy of the OpenMP
runtime, which LLVM libomp detects and aborts with
``OMP Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized``.

mdapy handles this in two ways depending on how it was installed:

- **From a PyPI wheel** (``pip install mdapy``) the wheel bundles its own
  OpenMP runtime so it works in a fresh environment. If you then import
  it alongside another package that bundles its own copy (commonly
  PyTorch) and hit the error above, set the flag yourself before
  importing any of the affected libraries:

  .. code-block:: python

     import os
     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

  or at the shell level:

  .. code-block:: bash

     export KMP_DUPLICATE_LIB_OK=TRUE

- **From source inside a conda env** (``pip install .`` with
  ``CONDA_PREFIX`` set) mdapy's build system detects the conda env and
  links against the OpenMP runtime that already lives there — the same
  one that conda-installed OVITO / PyTorch / freud are using — so only a
  single libomp ends up in the process and the conflict never arises.
  This is the recommended setup for development and for any environment
  that mixes these libraries heavily.

On macOS ``arm64`` specifically, OVITO and freud both bundle voro++ and
TBB, and importing both in one Python process crashes inside
``freud.locality.Voronoi.compute`` (or the equivalent OVITO modifier).
This is an upstream incompatibility — not something mdapy can fix. Run
mdapy's own test suite with ``./run_tests.sh`` (one subprocess per test
file) to sidestep it.

----

Quick Examples
--------------

**Load & analyse in three lines:**

.. code-block:: python

   import mdapy as mp

   sys = mp.System("fcc.dump")
   sys.cal_polyhedral_template_matching()   # adds 'ptm' column
   sys.cal_centro_symmetry_parameter(N=12)  # CSP for FCC
   print(sys.data)                          # Polars DataFrame

**Radial distribution function:**

.. code-block:: python

   rdf = sys.cal_radial_distribution_function(rc=6.0)
   rdf.plot()           # matplotlib figure — one line

**Machine-learning potential workflow:**

.. code-block:: python

   from mdapy import get_elastic_constant
   calc = mp.NEP('nep.txt')
   C = get_elastic_constant(sys, calc)
   C.print()   # 6×6 elastic tensor in GPa

**Ray-tracing render:**

.. code-block:: python

   from mdapy.render import TachyonRender

   ren = TachyonRender(backend="auto")          # GPU if available, else CPU
   ren.render_system(sys, width=1920, height=1080,
                     output_figure="structure.png")

**Build a polycrystal:**

.. code-block:: python

   unit = mp.build_crystal("Al", "fcc", 4.05)
   poly = mp.CreatePolycrystal(unit, box=100, seed_number=10, metal_overlap_dis=2.0)
   system = poly.compute()
   system.write_xyz("polycrystal.xyz")

----

Supported File Formats
----------------------

+------------------------+---------------------------+
| Format                 | Read / Write              |
+========================+===========================+
| LAMMPS DUMP            | ✅ / ✅                   |
+------------------------+---------------------------+
| LAMMPS DATA            | ✅ / ✅                   |
+------------------------+---------------------------+
| VASP POSCAR/CONTCAR    | ✅ / ✅                   |
+------------------------+---------------------------+
| XYZ (extended)         | ✅ / ✅                   |
+------------------------+---------------------------+
| MP (mdapy native)      | ✅ / ✅                   |
+------------------------+---------------------------+
| ASE Atoms              | ✅ (import / export)      |
+------------------------+---------------------------+
| OVITO DataCollection   | ✅ (import / export)      |
+------------------------+---------------------------+

----

Documentation & Resources
--------------------------

+----------------------------------+------------------------------------------------------+
| 📖 Full documentation            | https://mdapy.readthedocs.io                         |
+----------------------------------+------------------------------------------------------+
| 🏠 Source code                   | https://github.com/mushroomfire/mdapy                |
+----------------------------------+------------------------------------------------------+
| 🐛 Issue tracker                 | https://github.com/mushroomfire/mdapy/issues         |
+----------------------------------+------------------------------------------------------+

----

Dependencies
------------

**Required:**

- `numpy <https://numpy.org>`_ — array engine
- `polars <https://pola.rs>`_ — fast DataFrame backend

**Optional:**

- ``matplotlib`` — plotting (``pip install mdapy[plot]``)
- ``k3d`` — interactive 3-D notebook viewer (``pip install mdapy[k3d]``)
- ``pyfftw`` — faster FFT for MSD calculations
- ``phonopy`` — phonon dispersion
- ``lammps`` — LAMMPS-based potential calculations

----

Running the Tests
-----------------

Set up a dedicated environment with the reference libraries mdapy compares
against:

.. code-block:: bash

   conda create -n mda python=3.13
   conda activate mda
   conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito
   conda install -c conda-forge freud scipy scikit-learn pyfftw pymatgen ase
   pip install pytest
   pip install git+https://github.com/bigd4/PyNEP.git
   pip install .
   # also install LAMMPS via your preferred method

Then run the suite:

.. code-block:: bash

   bash run_tests.sh

``run_tests.sh`` invokes ``pytest`` once per test file. This is required on
macOS because ovito and freud both ship voro++ + TBB and cannot coexist in a
single Python process — an upstream incompatibility, not mdapy's.

To run a subset:

.. code-block:: bash

   ./run_tests.sh test_centro_symmetry_parameter.py test_voronoi.py

----

Citation
--------

If mdapy contributes to a scientific publication, please cite:

.. code-block:: bibtex

   @article{mdapy2023,
     title   = {mdapy: A flexible and efficient analysis software for
                molecular dynamics simulations},
     journal = {Computer Physics Communications},
     pages   = {108764},
     year    = {2023},
     issn    = {0010-4655},
     doi     = {10.1016/j.cpc.2023.108764},
     url     = {https://www.sciencedirect.com/science/article/pii/S0010465523001091},
     author  = {Yong-Chao Wu and Jian-Li Shao},
   }

----

Version Notice
--------------

mdapy 1.0 is a ground-up rewrite with a new, cleaner API.
The legacy release (≤ 0.11.5) is preserved on the
`mdapy_old branch <https://github.com/mushroomfire/mdapy/tree/mdapy_old>`_.

----

Contributing
------------

Bug reports, feature requests, and pull requests are all very welcome!
Please open an `issue <https://github.com/mushroomfire/mdapy/issues>`_ or
submit a PR on GitHub.

----

License
-------

BSD 3-Clause — see `LICENSE <https://github.com/mushroomfire/mdapy/blob/master/LICENSE>`_
for details.

----

Contact
-------

- **Issues / feature requests:** https://github.com/mushroomfire/mdapy/issues
- **Email:** 934313174@qq.com

If mdapy helps your research, a ⭐ on
`GitHub <https://github.com/mushroomfire/mdapy>`_ is always appreciated!
