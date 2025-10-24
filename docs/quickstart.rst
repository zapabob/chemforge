Quick Start
===========

This guide will help you get started with ChemForge quickly. We'll cover 
the basic concepts and show you how to use the main features.

Basic Concepts
==============

ChemForge is organized into several main modules:

* **Data Processing**: Molecular feature extraction and data management
* **Models**: Machine learning models for predictions
* **ADMET**: ADMET property predictions
* **Generation**: Molecular generation and optimization
* **GUI**: Web interfaces for interactive use

Your First Example
==================

Let's start with a simple example that demonstrates the core functionality:

.. code-block:: python

   from chemforge.data.molecular_features import MolecularFeatures
   from chemforge.models.transformer_model import TransformerModel
   from chemforge.admet.admet_predictor import ADMETPredictor
   
   # Define some molecules
   molecules = ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O']
   
   # Extract molecular features
   features_extractor = MolecularFeatures()
   features = features_extractor.extract_features(molecules)
   print(f"Extracted features shape: {features.shape}")
   
   # Make predictions
   model = TransformerModel(input_dim=features.shape[1], output_dim=5)
   predictions = model.predict(features)
   print(f"Predictions shape: {predictions.shape}")
   
   # Analyze ADMET properties
   admet_predictor = ADMETPredictor()
   admet_results = admet_predictor.predict_properties(molecules)
   print(f"ADMET results: {admet_results}")

Molecular Feature Extraction
============================

ChemForge provides comprehensive molecular feature extraction:

.. code-block:: python

   from chemforge.data.molecular_features import MolecularFeatures
   from chemforge.data.rdkit_descriptors import RDKitDescriptors
   
   # Basic molecular features
   features_extractor = MolecularFeatures()
   features = features_extractor.extract_features(['CCO', 'CCN'])
   
   # RDKit descriptors
   rdkit_descriptors = RDKitDescriptors()
   descriptors = rdkit_descriptors.calculate_descriptors(['CCO', 'CCN'])
   
   print(f"Basic features: {features.shape}")
   print(f"RDKit descriptors: {descriptors.shape}")

Model Predictions
=================

ChemForge supports multiple model architectures:

.. code-block:: python

   from chemforge.models.transformer_model import TransformerModel
   from chemforge.models.gnn_model import GNNModel
   from chemforge.models.ensemble_model import EnsembleModel
   
   # Transformer model
   transformer = TransformerModel(
       input_dim=200,
       output_dim=5,
       hidden_dim=256,
       num_heads=8,
       num_layers=4
   )
   
   # GNN model
   gnn = GNNModel(
       node_features=200,
       hidden_dim=256,
       output_dim=5,
       num_layers=3
   )
   
   # Ensemble model
   ensemble = EnsembleModel(
       models=[transformer, gnn],
       weights=[0.5, 0.5]
   )
   
   # Make predictions
   predictions = ensemble.predict(features)

ADMET Analysis
==============

ADMET analysis provides detailed molecular property predictions:

.. code-block:: python

   from chemforge.admet.admet_predictor import ADMETPredictor
   from chemforge.admet.property_predictor import PropertyPredictor
   from chemforge.admet.toxicity_predictor import ToxicityPredictor
   
   # ADMET predictions
   admet_predictor = ADMETPredictor()
   admet_results = admet_predictor.predict_properties(['CCO', 'CCN'])
   
   # Property predictions
   property_predictor = PropertyPredictor()
   properties = property_predictor.predict_properties(['CCO', 'CCN'])
   
   # Toxicity predictions
   toxicity_predictor = ToxicityPredictor()
   toxicity = toxicity_predictor.predict_toxicity(['CCO', 'CCN'])
   
   print(f"ADMET results: {admet_results}")
   print(f"Properties: {properties}")
   print(f"Toxicity: {toxicity}")

Molecular Generation
====================

ChemForge supports multiple molecular generation approaches:

.. code-block:: python

   from chemforge.generation.molecular_generator import MolecularGenerator
   
   # Initialize generator
   generator = MolecularGenerator()
   
   # Setup VAE generator
   vae_generator = generator.setup_vae_generator(
       input_dim=200,
       latent_dim=64,
       hidden_dim=256
   )
   
   # Generate molecules
   generated_molecules = generator.generate_with_vae(
       training_smiles=['CCO', 'CCN', 'CC(C)O'],
       num_molecules=10
   )
   
   print(f"Generated molecules: {generated_molecules}")

Web Interface
=============

ChemForge provides web interfaces for interactive use:

Streamlit Interface
-------------------

.. code-block:: bash

   streamlit run chemforge/gui/streamlit_app.py

Dash Interface
--------------

.. code-block:: bash

   python chemforge/gui/dash_app.py

Training Models
===============

To train your own models:

.. code-block:: python

   from chemforge.training.trainer import Trainer
   from chemforge.training.loss_functions import MSELoss
   from chemforge.training.metrics import RMSE, MAE
   
   # Initialize trainer
   trainer = Trainer(
       model=transformer,
       loss_function=MSELoss(),
       metrics=[RMSE(), MAE()],
       device='cpu'
   )
   
   # Train model
   trainer.train(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=100
   )

Data Management
==============

ChemForge provides comprehensive data management:

.. code-block:: python

   from chemforge.data.chembl_loader import ChEMBLLoader
   
   # Load ChEMBL data
   chembl_loader = ChEMBLLoader()
   data = chembl_loader.load_data(
       targets=['5-HT2A', 'D2R'],
       min_activities=100
   )
   
   print(f"Loaded {len(data)} compounds")

Visualization
=============

ChemForge includes comprehensive visualization tools:

.. code-block:: python

   from chemforge.utils.visualization import VisualizationUtils
   
   # Create visualizations
   viz = VisualizationUtils()
   
   # Molecular properties plot
   viz.plot_molecular_properties(data)
   
   # ADMET radar chart
   viz.plot_admet_radar(admet_results)
   
   # Prediction results
   viz.plot_predictions(predictions)

Next Steps
==========

Now that you have a basic understanding of ChemForge, you can:

1. **Explore the User Guide**: Detailed documentation for each module
2. **Check out Examples**: More complex usage examples
3. **Read the API Reference**: Complete API documentation
4. **Join the Community**: Get help and contribute

For more information, see the :doc:`user_guide/index` section.