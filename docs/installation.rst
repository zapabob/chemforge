Installation
============

ChemForge can be installed using pip or from source. This section covers 
different installation methods and requirements.

Requirements
============

ChemForge requires Python 3.8 or higher and the following dependencies:

* NumPy >= 1.21.0
* Pandas >= 1.3.0
* PyTorch >= 1.9.0
* Scikit-learn >= 1.0.0
* RDKit >= 2022.03.0
* Plotly >= 5.0.0
* Streamlit >= 1.0.0
* Dash >= 2.0.0

Installation Methods
====================

PyPI Installation
-----------------

The easiest way to install ChemForge is using pip:

.. code-block:: bash

   pip install chemforge

For the latest development version:

.. code-block:: bash

   pip install chemforge --upgrade

Conda Installation
------------------

ChemForge can also be installed using conda:

.. code-block:: bash

   conda install -c conda-forge chemforge

Source Installation
-------------------

To install from source:

.. code-block:: bash

   git clone https://github.com/zapabob/chemforge.git
   cd chemforge
   pip install -e .

Development Installation
========================

For development, install in editable mode with development dependencies:

.. code-block:: bash

   git clone https://github.com/zapabob/chemforge.git
   cd chemforge
   pip install -e ".[dev]"

This will install ChemForge in editable mode along with development 
dependencies like pytest, black, flake8, and sphinx.

Verification
===========

To verify the installation, run:

.. code-block:: python

   import chemforge
   print(chemforge.__version__)

Or run the test suite:

.. code-block:: bash

   pytest tests/

Docker Installation
===================

ChemForge can also be run using Docker:

.. code-block:: bash

   docker pull chemforge/chemforge:latest
   docker run -p 8501:8501 chemforge/chemforge:latest

This will start the Streamlit web interface on port 8501.

Troubleshooting
===============

Common Installation Issues
--------------------------

**RDKit Installation Issues**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RDKit can be tricky to install. If you encounter issues:

.. code-block:: bash

   conda install -c conda-forge rdkit

**PyTorch Installation Issues**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For PyTorch installation issues, visit the official PyTorch website 
for platform-specific installation instructions.

**CUDA Support**
~~~~~~~~~~~~~~~~

For CUDA support, install PyTorch with CUDA:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

**Memory Issues**
~~~~~~~~~~~~~~~~

If you encounter memory issues, consider:

* Using smaller batch sizes
* Enabling gradient checkpointing
* Using mixed precision training

Platform-Specific Notes
=======================

Windows
-------

On Windows, you may need to install Visual C++ Build Tools for some 
dependencies. Use the conda installation method if possible.

macOS
-----

On macOS, you may need to install Xcode command line tools:

.. code-block:: bash

   xcode-select --install

Linux
-----

On Linux, you may need to install development headers:

.. code-block:: bash

   sudo apt-get install python3-dev build-essential

Uninstallation
==============

To uninstall ChemForge:

.. code-block:: bash

   pip uninstall chemforge

Or if installed with conda:

.. code-block:: bash

   conda remove chemforge