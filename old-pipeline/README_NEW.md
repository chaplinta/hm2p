# HM2P-Analysis

A neuroscience analysis pipeline for processing simultaneous two-photon calcium imaging and behavioral tracking data from freely moving mice.

## Overview

HM2P integrates:
- **Calcium imaging data** from Suite2p (two-photon microscopy)
- **Behavioral tracking** from DeepLabCut (pose estimation)
- **Hardware synchronization** from DAQ systems

The pipeline enables correlation of neural activity patterns with behavioral states during spatial navigation tasks.

## Key Features

### Neural Analysis
- Delta-F/F0 calculation with neuropil correction
- Calcium event detection using probabilistic thresholding
- Spike deconvolution
- Soma vs. dendrite signal separation
- Signal-to-noise quantification

### Behavioral Metrics
- Head direction (absolute, filtered, unwrapped)
- Angular head velocity (instantaneous and gradient-based)
- Position tracking (pixels, mm, maze coordinates)
- Locomotion speed and acceleration
- Allocentric vs. egocentric heading

### Tuning Analysis
- Head direction (HD) tuning curves
- Angular head velocity (AHV) tuning
- Place field analysis
- Speed selectivity
- Circular statistics (Rayleigh test, preferred direction)
- Bootstrap confidence intervals
- Cross-session stability analysis

## Project Structure

```
hm2p-analysis/
├── classes/           # Core data classes
│   ├── Experiment.py  # Master experiment metadata handler
│   ├── S2PData.py     # Suite2p data management
│   ├── ProcPath.py    # Path management
│   └── TrackingCamera.py
├── proc/              # Processing modules
│   ├── proc_ca.py     # Calcium signal processing
│   ├── proc_behave.py # Behavioral metric calculation
│   ├── proc_ca_behave.py  # Synchronization & resampling
│   ├── proc_somadend.py   # Soma-dendrite pair analysis
│   ├── proc_s2p.py    # Suite2p pipeline
│   └── proc_brainreg.py   # Brain registration
├── utils/             # Utility modules
│   ├── tune.py        # Tuning curve analysis
│   ├── ca.py          # Calcium utilities
│   ├── behave.py      # Behavioral calculations
│   ├── db.py          # Database management
│   └── stats.py       # Statistical tests
├── sum/               # Summary/aggregation
│   ├── sum_tune.py    # Tuning curve summaries
│   ├── sum_ca_behave.py
│   ├── sum_behave.py
│   ├── sum_events.py
│   └── sum_somadend.py
├── notebooks/         # Analysis notebooks
└── run.py             # Main entry point
```

## Data Flow

```
Raw Data (2p images, videos, DAQ)
         │
         ▼
┌─────────────────────────────────┐
│  Preprocessing                  │
│  - Suite2p ROI extraction       │
│  - DeepLabCut pose tracking     │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Signal Processing              │
│  - dF/F0 calculation            │
│  - Event detection              │
│  - Behavioral metrics           │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Synchronization                │
│  - Align neural to behavior     │
│  - Resample to common timebase  │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Analysis                       │
│  - Tuning curves                │
│  - Population statistics        │
│  - Soma-dendrite comparisons    │
└─────────────────────────────────┘
```

## Usage

```python
from classes.Experiment import Experiment

# Load experiment
exp = Experiment('path/to/experiment')

# Run processing pipeline
from proc import proc_ca, proc_behave, proc_ca_behave

proc_ca.process(exp)
proc_behave.process(exp)
proc_ca_behave.process(exp)
```

## Dependencies

- numpy
- pandas
- scipy
- matplotlib
- suite2p
- DeepLabCut
