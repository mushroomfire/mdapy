Installation
=============

Method 1 — pip install (recommended)
------------------------------------

.. code-block:: bash

   pip install mdapy


.. note::

   **GPU-accelerated ray-tracing (NVIDIA OptiX 7)**

   Pre-compiled wheels with GPU rendering support are published for
   **Linux x86_64** and **Windows x86_64**. The GPU wheel is selected
   automatically by pip on supported platforms:

   - **Linux** (glibc ≥ 2.28, e.g. Ubuntu 20.04+): pip automatically installs
     the GPU-capable wheel. Older Linux systems receive the CPU-only wheel.
   - **Windows**: the published wheel always includes GPU support.
   - **macOS**: CPU-only (Apple Silicon does not support CUDA).

   The GPU wheel works on CPU-only machines too — ``backend='gpu'`` will raise
   a ``RuntimeError`` if no suitable GPU is found, while ``backend='cpu'`` and
   all other features continue to work normally.

   **Runtime requirements (GPU features only):**

   - NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal or newer)
   - NVIDIA display driver ≥ 520 (ships with OptiX 7 support)
   - No CUDA Toolkit installation needed — the runtime is statically linked

   Check GPU availability at runtime:

   .. code-block:: python

      from mdapy import render
      print(render.is_gpu_available())   # True if GPU backend is usable


Method 2 — install with optional extras
---------------------------------------

.. code-block:: bash

   pip install mdapy[plot]      # + matplotlib
   pip install mdapy[k3d]       # + k3d 3-D widget
   pip install mdapy[all]       # + matplotlib, k3d, pyfftw, phonopy

LAMMPS is not an extra — install it via your preferred channel
(conda-forge, system package manager, or build from source).


Method 3 — build from the PyPI source distribution
--------------------------------------------------

If no pre-built wheel fits your platform, or you want to compile with
your own compiler flags, install from the **sdist** also published to
PyPI. The sdist contains only the C++ sources, Python package and build
configuration (no git history, no docs, no test input files), and pip
downloads it automatically:

.. code-block:: bash

   pip install --no-binary mdapy mdapy

The ``--no-binary mdapy`` flag skips the wheel for ``mdapy`` only —
``numpy`` and ``polars`` still install as wheels.


Method 4 — install from a git clone (for development)
-----------------------------------------------------

.. code-block:: bash

   git clone https://github.com/mushroomfire/mdapy.git
   cd mdapy
   pip install .          # regular install
   # or:
   pip install -e .       # editable, picks up source changes without reinstall

Use this if you intend to modify the source, run the tests, or build
the documentation.


Method 5 — install directly from GitHub
---------------------------------------

.. code-block:: bash

   pip install https://github.com/mushroomfire/mdapy/archive/master.zip


System requirements for Methods 3 – 5
-------------------------------------

All three compile mdapy's C++ extensions locally and therefore need:

- Python 3.9+
- C++17-capable compiler
- OpenMP support
- Tested platforms: Windows 10 (MSVC), Ubuntu (GCC), macOS Apple Silicon (Clang)


Verify the installation
-----------------------

.. code-block:: bash

   python -c "import mdapy as mp; print('mdapy', mp.__version__, '— ready!')"


Interoperability with PyTorch / OVITO / freud / scikit-learn
------------------------------------------------------------

mdapy, PyTorch, OVITO, freud and scikit-learn all use OpenMP for internal
parallelism. Because each project ships its own pre-built binaries, a
single Python process can end up with more than one copy of the OpenMP
runtime, which LLVM libomp detects and aborts with::

   OMP Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.

mdapy handles this in two ways depending on how it was installed:

**From a PyPI wheel** (``pip install mdapy``) the wheel bundles its own
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

**From source inside a conda env** (``pip install .`` with
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
