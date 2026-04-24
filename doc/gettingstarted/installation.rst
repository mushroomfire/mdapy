Installation
=============

**Method 1: Install via pip (Recommended)**

.. code-block:: bash

   pip install mdapy


.. note::

   **GPU-accelerated ray-tracing (NVIDIA OptiX 7)**

   Pre-compiled wheels with GPU rendering support are published for
   **Linux x86_64** and **Windows x86_64**.  The GPU wheel is selected
   automatically by pip on supported platforms:

   - **Linux** (glibc ≥ 2.28, e.g. Ubuntu 20.04+): pip automatically installs
     the GPU-capable wheel.  Older Linux systems receive the CPU-only wheel.
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


**Method 2: Install with all optional features**

.. code-block:: bash

   # Install all optional packages (matplotlib, k3d, phonopy, pyfftw), except lammps
   pip install mdapy[all]

**Method 3: Build from the PyPI source distribution**

If no pre-built wheel fits your platform, or you want to compile with
your own compiler flags, install from the **sdist** we also publish to
PyPI. This is the smallest way to build from source — the sdist contains
only the C++ sources, Python package, and build configuration (no git
history, no documentation, no test input files), and pip downloads it
for you automatically:

.. code-block:: bash

   pip install --no-binary mdapy mdapy

The ``--no-binary mdapy`` flag tells pip to skip the wheel and fetch the
sdist from PyPI. Dependencies (``numpy``, ``polars``) are still installed
as wheels — only ``mdapy`` itself is built locally.

**Method 4: Install from a git clone (for development)**

.. code-block:: bash

   git clone https://github.com/mushroomfire/mdapy.git
   cd mdapy
   pip install .

Use this if you intend to modify the source, run the tests, or build the
documentation. For an editable install (source changes are picked up
without reinstalling) use ``pip install -e .``.

**Method 5: Install directly from GitHub**

.. code-block:: bash

   pip install https://github.com/mushroomfire/mdapy/archive/master.zip

**System Requirements for Methods 3 – 5**

All three compile mdapy's C++ extensions locally and therefore need:

- Python 3.8+
- C++ compilation environment (C++17 or newer)
- OpenMP support
- Tested platforms: Windows 10 (MSVC), Ubuntu (GCC), Mac OS M1 (Clang)

**Verify Installation**

.. code-block:: bash

   python -c "import mdapy as mp; print('mdapy version is:', mp.__version__)"
