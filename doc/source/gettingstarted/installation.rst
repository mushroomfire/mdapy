Installation
==============

1. Download source code
   
   .. code-block:: bash

      git clone https://github.com/mushroomfire/mdapy.git
      cd mdapy 

2. Create a virtual environment (highly recommoned using anaconda)
   
   - It helps you to avoid the potential conflicts of dependencies in your current python environment and is very suitable for beginners. If you have installed all dependencies listed above, you can just skip this step!
   
   .. code-block:: bash

      conda create -n mda python==3.8.0 # python version larger than 3.7 and lower than 3.11 is okay.
      conda activate mda

3. Install dependencies
   
   .. code-block:: bash

      pip install -r requirements.txt

4. Install mdapy
   
   .. code-block:: bash

      python install.py

5. Update mdapy
   
   - Just download the latest souce code and repeat the step 3 and step 4.