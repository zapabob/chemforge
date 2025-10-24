# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Molecular PWA+PET Transformer library
- Core PWA+PET Transformer implementation
- CNS targets support (D1, CB1/CB2, opioid receptors)
- Molecular generation and QSAR prediction
- Docking integration (EquiBind, DiffDock, GNINA)
- ADMET evaluation (SwissADME, pkCSM, CNS-MPO)
- CLI interface for training, generation, and evaluation
- Comprehensive test suite
- Documentation and examples

### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

## [1.0.0] - 2025-01-XX

### Added
- **Core Features**
  - PWA+PET Transformer architecture
  - Group-equivariant attention mechanisms
  - Phase gates (SU(2) unitary transformations)
  - Multi-task learning (pKi, activity, CNS-MPO, QED, SA)
  
- **CNS Targets**
  - Serotonin receptors (5-HT2A, 5-HT1A)
  - Dopamine receptors (D1, D2)
  - Cannabinoid receptors (CB1, CB2)
  - Opioid receptors (MOR, DOR, KOR, NOP)
  - Transporters (SERT, DAT, NET)
  
- **Molecular Processing**
  - SMILES/SELFIES support
  - 3D molecular coordinates
  - Atomic features and bond features
  - Molecular graph representation
  
- **Training & Evaluation**
  - Multi-task loss functions
  - CNS-specific metrics
  - Hyperparameter optimization
  - Model checkpointing
  
- **Generation**
  - VAE-based molecular generation
  - RNN-based molecular generation
  - Transformer-based molecular generation
  - Goal-directed optimization
  
- **Docking Integration**
  - EquiBind wrapper
  - DiffDock wrapper
  - GNINA wrapper
  - Pose scoring and ranking
  
- **ADMET Evaluation**
  - SwissADME integration
  - pkCSM integration
  - CNS-MPO calculation
  - QED and SA scoring
  
- **CLI Interface**
  - Training command
  - Generation command
  - Evaluation command
  - Configuration management
  
- **Testing**
  - Unit tests for core functionality
  - Integration tests for pipelines
  - Performance benchmarks
  - Memory usage tests
  
- **Documentation**
  - API documentation
  - Tutorial notebooks
  - Usage examples
  - Configuration guides

### Changed
- None (initial release)

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

## [0.1.0] - 2025-01-XX (Development)

### Added
- **Development Setup**
  - Project structure
  - Package configuration
  - Development dependencies
  - Pre-commit hooks
  - CI/CD pipeline
  
- **Core Implementation**
  - Basic Transformer architecture
  - PWA attention mechanism
  - PET phase gates
  - Multi-task heads
  
- **Data Processing**
  - Molecular dataset classes
  - Preprocessing pipelines
  - Data augmentation
  - Feature extraction
  
- **Model Training**
  - Training loops
  - Loss functions
  - Optimizers
  - Schedulers
  
- **Evaluation**
  - Metrics calculation
  - Visualization
  - Reporting
  - Benchmarking

### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None
