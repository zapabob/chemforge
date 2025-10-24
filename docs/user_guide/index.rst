User Guide
==========

This section provides comprehensive user documentation for ChemForge.

.. toctree::
   :maxdepth: 2

   data_processing
   models
   admet
   generation
   training
   gui
   visualization
   data_management
   troubleshooting

Data Processing
===============

ChemForge provides comprehensive data processing capabilities for molecular data.

Molecular Feature Extraction
-----------------------------

The :class:`MolecularFeatures` class provides molecular feature extraction:

.. code-block:: python

   from chemforge.data.molecular_features import MolecularFeatures
   
   # Initialize feature extractor
   features_extractor = MolecularFeatures()
   
   # Extract features
   molecules = ['CCO', 'CCN', 'CC(C)O']
   features = features_extractor.extract_features(molecules)
   
   print(f"Features shape: {features.shape}")

RDKit Descriptors
-----------------

The :class:`RDKitDescriptors` class provides RDKit-based molecular descriptors:

.. code-block:: python

   from chemforge.data.rdkit_descriptors import RDKitDescriptors
   
   # Initialize descriptor calculator
   rdkit_descriptors = RDKitDescriptors()
   
   # Calculate descriptors
   descriptors = rdkit_descriptors.calculate_descriptors(molecules)
   
   print(f"Descriptors shape: {descriptors.shape}")

Data Preprocessing
------------------

The :class:`DataPreprocessor` class provides data preprocessing capabilities:

.. code-block:: python

   from chemforge.data.data_preprocessor import DataPreprocessor
   
   # Initialize preprocessor
   preprocessor = DataPreprocessor()
   
   # Preprocess data
   processed_data = preprocessor.preprocess_data(data)
   
   print(f"Processed data shape: {processed_data.shape}")

Models
======

ChemForge supports multiple machine learning model architectures.

Transformer Models
------------------

The :class:`TransformerModel` class provides transformer-based models:

.. code-block:: python

   from chemforge.models.transformer_model import TransformerModel
   
   # Initialize transformer
   transformer = TransformerModel(
       input_dim=200,
       output_dim=5,
       hidden_dim=256,
       num_heads=8,
       num_layers=4
   )
   
   # Make predictions
   predictions = transformer.predict(features)

GNN Models
----------

The :class:`GNNModel` class provides graph neural network models:

.. code-block:: python

   from chemforge.models.gnn_model import GNNModel
   
   # Initialize GNN
   gnn = GNNModel(
       node_features=200,
       hidden_dim=256,
       output_dim=5,
       num_layers=3
   )
   
   # Make predictions
   predictions = gnn.predict(features)

Ensemble Models
---------------

The :class:`EnsembleModel` class provides ensemble learning:

.. code-block:: python

   from chemforge.models.ensemble_model import EnsembleModel
   
   # Initialize ensemble
   ensemble = EnsembleModel(
       models=[transformer, gnn],
       weights=[0.5, 0.5]
   )
   
   # Make predictions
   predictions = ensemble.predict(features)

ADMET Analysis
==============

ChemForge provides comprehensive ADMET analysis capabilities.

ADMET Predictor
---------------

The :class:`ADMETPredictor` class provides ADMET property predictions:

.. code-block:: python

   from chemforge.admet.admet_predictor import ADMETPredictor
   
   # Initialize ADMET predictor
   admet_predictor = ADMETPredictor()
   
   # Predict properties
   properties = admet_predictor.predict_properties(molecules)
   
   print(f"ADMET properties: {properties}")

Property Predictor
------------------

The :class:`PropertyPredictor` class provides molecular property predictions:

.. code-block:: python

   from chemforge.admet.property_predictor import PropertyPredictor
   
   # Initialize property predictor
   property_predictor = PropertyPredictor()
   
   # Predict properties
   properties = property_predictor.predict_properties(molecules)
   
   print(f"Properties: {properties}")

Toxicity Predictor
------------------

The :class:`ToxicityPredictor` class provides toxicity predictions:

.. code-block:: python

   from chemforge.admet.toxicity_predictor import ToxicityPredictor
   
   # Initialize toxicity predictor
   toxicity_predictor = ToxicityPredictor()
   
   # Predict toxicity
   toxicity = toxicity_predictor.predict_toxicity(molecules)
   
   print(f"Toxicity: {toxicity}")

Molecular Generation
====================

ChemForge supports multiple molecular generation approaches.

VAE Generation
--------------

The :class:`VAEGenerator` class provides VAE-based molecular generation:

.. code-block:: python

   from chemforge.generation.vae_generator import VAEGenerator
   
   # Initialize VAE generator
   vae_generator = VAEGenerator(
       input_dim=200,
       latent_dim=64,
       hidden_dim=256
   )
   
   # Generate molecules
   generated_molecules = vae_generator.generate_molecules(num_molecules=10)
   
   print(f"Generated molecules: {generated_molecules}")

RL Optimization
---------------

The :class:`RLOptimizer` class provides reinforcement learning-based optimization:

.. code-block:: python

   from chemforge.generation.rl_optimizer import RLOptimizer
   
   # Initialize RL optimizer
   rl_optimizer = RLOptimizer(
       state_dim=200,
       action_dim=10,
       hidden_dim=128
   )
   
   # Optimize molecules
   optimized_molecules = rl_optimizer.optimize_molecules(
       initial_molecules=molecules,
       reward_function=reward_function
   )
   
   print(f"Optimized molecules: {optimized_molecules}")

Genetic Algorithm
-----------------

The :class:`GeneticOptimizer` class provides genetic algorithm-based optimization:

.. code-block:: python

   from chemforge.generation.genetic_optimizer import GeneticOptimizer
   
   # Initialize genetic optimizer
   genetic_optimizer = GeneticOptimizer(
       population_size=100,
       mutation_rate=0.1,
       crossover_rate=0.8
   )
   
   # Optimize molecules
   optimized_molecules = genetic_optimizer.optimize(
       initial_molecules=molecules,
       fitness_function=fitness_function
   )
   
   print(f"Optimized molecules: {optimized_molecules}")

Training
========

ChemForge provides comprehensive training capabilities.

Trainer
-------

The :class:`Trainer` class provides model training:

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

Loss Functions
---------------

ChemForge provides various loss functions:

.. code-block:: python

   from chemforge.training.loss_functions import MSELoss, MAELoss, HuberLoss
   
   # Initialize loss functions
   mse_loss = MSELoss()
   mae_loss = MAELoss()
   huber_loss = HuberLoss()
   
   # Use in training
   trainer = Trainer(
       model=transformer,
       loss_function=mse_loss,
       metrics=[RMSE(), MAE()]
   )

Metrics
-------

ChemForge provides various evaluation metrics:

.. code-block:: python

   from chemforge.training.metrics import RMSE, MAE, R2, Pearson
   
   # Initialize metrics
   rmse = RMSE()
   mae = MAE()
   r2 = R2()
   pearson = Pearson()
   
   # Use in training
   trainer = Trainer(
       model=transformer,
       loss_function=MSELoss(),
       metrics=[rmse, mae, r2, pearson]
   )

GUI
===

ChemForge provides web interfaces for interactive use.

Streamlit Interface
-------------------

The Streamlit interface provides an intuitive web interface:

.. code-block:: bash

   streamlit run chemforge/gui/streamlit_app.py

Features include:

* Molecular analysis and visualization
* AI predictions and ADMET analysis
* Molecular generation capabilities
* Data management and export

Dash Interface
--------------

The Dash interface provides a more flexible web interface:

.. code-block:: bash

   python chemforge/gui/dash_app.py

Features include:

* Customizable layouts
* Interactive visualizations
* Real-time updates
* Advanced styling

Visualization
=============

ChemForge provides comprehensive visualization capabilities.

Visualization Utils
-------------------

The :class:`VisualizationUtils` class provides visualization utilities:

.. code-block:: python

   from chemforge.utils.visualization import VisualizationUtils
   
   # Initialize visualization utils
   viz = VisualizationUtils()
   
   # Create visualizations
   viz.plot_molecular_properties(data)
   viz.plot_admet_radar(admet_results)
   viz.plot_predictions(predictions)

Data Management
===============

ChemForge provides comprehensive data management capabilities.

ChEMBL Integration
------------------

The :class:`ChEMBLLoader` class provides ChEMBL database integration:

.. code-block:: python

   from chemforge.data.chembl_loader import ChEMBLLoader
   
   # Initialize ChEMBL loader
   chembl_loader = ChEMBLLoader()
   
   # Load data
   data = chembl_loader.load_data(
       targets=['5-HT2A', 'D2R'],
       min_activities=100
   )
   
   print(f"Loaded {len(data)} compounds")

Troubleshooting
===============

Common Issues and Solutions
---------------------------

**Memory Issues**
~~~~~~~~~~~~~~~~

If you encounter memory issues:

* Use smaller batch sizes
* Enable gradient checkpointing
* Use mixed precision training

**Performance Issues**
~~~~~~~~~~~~~~~~~~~~~~

For performance optimization:

* Use GPU acceleration
* Enable model compilation
* Use optimized data loaders

**Installation Issues**
~~~~~~~~~~~~~~~~~~~~~~~

For installation problems:

* Check Python version compatibility
* Install dependencies in order
* Use virtual environments

Getting Help
============

If you need help:

1. Check the documentation
2. Search existing issues
3. Create a new issue
4. Join the community
