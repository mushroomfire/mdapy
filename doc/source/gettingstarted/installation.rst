Installation
=============

Install from pip (recommended).
--------------------------------------

.. code-block:: bash

   pip install mdapy

Install from source code.
--------------------------------------

- You should install pybind11 and have a C++ compilation environment (-std=c++11 or newer) and openmp supports. 
  Tested by MSVC in Windows 10, GCC in Ubuntu, Clang in MAC OS M1.

   .. code-block:: bash
      
      pip install pybind11

- Download the source code and installation.
   
   .. code-block:: bash

      git clone https://github.com/mushroomfire/mdapy.git
      cd mdapy 
      pip install .

- One may install openmp in MAC OS manually by: 
   
   .. code-block:: bash 

      curl -O https://mac.r-project.org/openmp/openmp-13.0.0-darwin21-Release.tar.gz 
      sudo tar fvxz openmp-13.0.0-darwin21-Release.tar.gz -C /
   
- I have installed Xcode in my laptop, so the compiler args should be: "*-Xclang -fopenmp*".
  If you install openmp by: *brew install libomp*, you can try to change the compiler args to "*-fopenmp*".
