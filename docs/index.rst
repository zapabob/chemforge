ChemForge Documentation
========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide/index
   api_reference/index
   examples/index
   development/index
   changelog

Welcome to ChemForge!
=====================

ChemForge is a cutting-edge platform for CNS drug discovery that combines 
advanced machine learning models with intuitive web interfaces to accelerate 
drug discovery research.

Key Features
============

* **Advanced Molecular Analysis**: Comprehensive molecular feature extraction and analysis
* **AI-Powered Predictions**: Multi-target pIC50 predictions using state-of-the-art models
* **ADMET Analysis**: Detailed ADMET property predictions and CNS-MPO scoring
* **Molecular Generation**: VAE, reinforcement learning, and genetic algorithm-based molecular generation
* **Interactive Web Interface**: Streamlit and Dash-based web applications
* **Comprehensive Data Management**: ChEMBL database integration and custom dataset support

Quick Start
===========

.. code-block:: python

   from chemforge.data.molecular_features import MolecularFeatures
   from chemforge.models.transformer_model import TransformerModel
   from chemforge.admet.admet_predictor import ADMETPredictor
   
   # Extract molecular features
   features_extractor = MolecularFeatures()
   features = features_extractor.extract_features(['CCO', 'CCN', 'CC(C)O'])
   
   # Make predictions
   model = TransformerModel(input_dim=200, output_dim=5)
   predictions = model.predict(features)
   
   # Analyze ADMET properties
   admet_predictor = ADMETPredictor()
   admet_results = admet_predictor.predict_properties(['CCO', 'CCN', 'CC(C)O'])

Installation
============

.. code-block:: bash

   pip install chemforge

For development installation:

.. code-block:: bash

   git clone https://github.com/zapabob/chemforge.git
   cd chemforge
   pip install -e .

Documentation Structure
=======================

This documentation is organized into several sections:

* **Installation**: How to install ChemForge
* **Quick Start**: Get up and running quickly
* **User Guide**: Comprehensive user documentation
* **API Reference**: Detailed API documentation
* **Examples**: Usage examples and tutorials
* **Development**: Developer documentation
* **Changelog**: Version history and changes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`