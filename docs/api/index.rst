API Reference
=============

This section contains the complete API reference for Molecular PWA+PET Transformer.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   core/transformer
   core/attention
   core/su2_gates
   core/utils

Data Processing
---------------

.. toctree::
   :maxdepth: 2

   data/molecular_dataset
   data/preprocessing
   data/augmentation

Models
------

.. toctree::
   :maxdepth: 2

   models/qsar_model
   models/generation_model
   models/scoring_model

Training
--------

.. toctree::
   :maxdepth: 2

   training/trainer
   training/losses
   training/metrics

Generation
----------

.. toctree::
   :maxdepth: 2

   generation/vae_generator
   generation/rnn_generator
   generation/transformer_generator

Docking
--------

.. toctree::
   :maxdepth: 2

   docking/equibind_wrapper
   docking/diffdock_wrapper
   docking/gnina_wrapper

ADMET
------

.. toctree::
   :maxdepth: 2

   admet/swissadme
   admet/pkcsmlab
   admet/cns_mpo

Targets
--------

.. toctree::
   :maxdepth: 2

   targets/cns_targets
   targets/target_utils

Pipeline
--------

.. toctree::
   :maxdepth: 2

   pipeline/molecular_pipeline
   pipeline/optimization

CLI
---

.. toctree::
   :maxdepth: 2

   cli/main
   cli/train
   cli/generate
   cli/evaluate
