Installation
=============

Install from pip (recommended).
--------------------------------------

.. code-block:: bash

   pip install mdapy

Install from source code.
---------------------------

- You should install pybind11 and have a c++ compilation environment. Tested by MSVC in Windows and GCC in Ubuntu.

   .. code-block:: bash
      
      pip install pybind11

- Download source code
   
   .. code-block:: bash

      git clone https://github.com/mushroomfire/mdapy.git
      cd mdapy 
      python setup.py install