# AI Negatively Impacts Team Performance

This repository contains analysis scripts for a comprehensive study comparing human-only teams versus human-AI collaborative teams across multiple behavioral, physiological, and performance measures. 

## Overview

The research examines team dynamics and performance differences between:
- **Human-only teams**: Teams consisting three human participants
- **Human-AI teams**: Teams where two human participants collaborate with an Wizard-of-Oz (WoZ) AI agent ("Alice")

## Study Design

### Team Roles
Each team consists of three control roles:
- **ThrustPilot**: Controls speed of movement
- **YawPilot**: Controls left/right movement  
- **PitchPilot**: Controls up/down movement

### Experimental Conditions
- **Sessions**: 3 sessions per team (S1, S2, S3)
- **Difficulty Levels**: Easy, Medium, Hard
- **Communication Types**: Incommunicado (no communication), Command word, Free Speech(free communication)
- **Performance Metric**: Number of rings passed through

## Data Sources

The analysis incorporates multiple data modalities:

### Behavioral Data
- **Controller Actions**: Input frequency across different control roles
- **Speech Communication**: Voice activity detection, speech frequency, and duration
- **Performance Metrics**: Task completion scores across different conditions

### Physiological Data
- **EEG (Electroencephalography)**: 20-channel EEG recording of YawPilots and PitchPilots
  - Inter-brain synchrony analysis for team neural synchronization. Inter-brain synchrony is measured by Total Interdependence (TI) 
  - Baseline-corrected TI measurements
- **Eye Tracking**: 
  - Pupil size measurements and percent change from baseline
  - Blink frequency analysis

### Subjective Measures
- **Questionnaires**: Helpfulness and leadership ratings across sessions for human ThrustPilot or the WoZ AI

## Analysis Scripts

### `compare_controller_action.py`
Analyzes controller input patterns across team types and roles.

**Key Functions:**
- `get_number_of_action()`: Counts continuous action chunks
- `plot_compaired_actions_all()`: Violin plots comparing action frequency
- Statistical testing with t-tests across roles (Thrust, Yaw, Pitch)

**Outputs:** 
- `plots/actions_all.png`: Action frequency comparison figure

### `compare_eeg_TI.py`
Computes and compares Inter-brain synchrony (TI) from EEG data as a measure of team neural synchronization.

**Key Functions:**
- `baseline_TI()`: Computes baseline TI over initial time window
- `total_interdependence_broadband()`: Calculates TI using coherence analysis
- `compute_ring_TI()`: Event-locked TI analysis
- `plot_TI_across_all_condition()`: Overall TI comparison
- `plot_TI_sess()`: Session-wise TI analysis

**Outputs:**
- `plots/eeg_plots/TI_total.png`: Overall TI comparison figure
- `plots/eeg_plots/TI_session.png`: Session-wise TI comparison figure

### `compare_eye_data.py`
Analyzes pupil size and blink patterns as indicators of cognitive load and attention.

**Key Functions:**
- `compute_percent_change()`: Calculates pupil size change from baseline
- `plot_pupil_percent_changes_all_conditions()`: Time-series pupil analysis with FDR correction
- `plot_pupil_size_session()`: Session-wise pupil size comparison
- `plot_number_of_blink_all_conditions()`: Blink frequency analysis

**Outputs:**
- `plots/pupil_size_comparison.png`: Pupil size time-series comparison figure
- `plots/pupil_sess.png`: Session-wise pupil analysis figure
- `plots/number_blink_all.png`: Overall blink frequency figure
- `plots/number_blink_sess_role.png`: Session-wise blink analysis figure

### `compare_performance.py`
Evaluates task performance across conditions and team types.

**Key Functions:**
- `plot_performance_all()`: Overall performance comparison
- `get_difficulty_performance()`: Performance by difficulty level
- `get_communication_performance()`: Performance by communication type
- Repeated measures ANOVA for within-subjects comparisons

**Outputs:**
- `plots/performance_all.png`: Overall performance comparison figure
- `plots/performance_difficulty.png`: Difficulty-based performance figure
- `plots/performance_communication.png`: Communication-based performance figure

### `compare_questionnaire.py`
Analyzes subjective ratings of ThrustPilot's leadership and helpfulness.

**Key Functions:**
- `plot_thrust_helpfulness()`: Helpfulness ratings across sessions
- `plot_thrust_leader()`: Leadership ratings across sessions
- `repeated_measure_ANOVA()`: Statistical analysis of rating changes

**Outputs:**
- `plots/helpfulness_thrust.png`: Helpfulness ratings figure
- `plots/leader_thrust.png`: Leadership ratings figure

### `compare_speech_event.py`
Processes and analyzes speech communication patterns.

**Key Functions:**
- `simple_vad()`: Voice activity detection with gap merging
- `process_speech_event_from_raw_speech_epochs()`: Speech event extraction
- `plot_communication_across_all_conditions()`: Communication frequency vs duration analysis

**Outputs:**
- `plots/comm_all.png`: Communication pattern comparison figure

### `supp_xxx.py`
Analyzes and results that included in supplementary material. 



## Data Structure

### Expected Data Files
```
data/
├── all_human/
│   ├── epoched_data/
│   │   ├── epoched_action.pkl
│   │   ├── epoched_pupil.pkl
│   │   └── epoched_raw_speech.pkl
│   ├── trialed_data/
│   │   └── trialed_numb_blink.pkl
│   ├── preprocessed_eeg.pkl
│   └── pupil_baseline.pkl
├── human_ai/
│   ├── epoched_data/
│   │   ├── epoched_action.pkl
│   │   ├── epoched_pupil.pkl
│   │   └── epoched_raw_speech.pkl
│   ├── trialed_data/
│   │   └── trialed_numb_blink.pkl
│   ├── preprocessed_eeg.pkl
│   └── pupil_baseline.pkl
├── eeg_epochs_both_human_and_alice.pkl
├── speech_epochs_both_human_and_alice.pkl
├── performance.pkl
└── helpfulness_leadership.csv
```

## Requirements

### Python Dependencies
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import scipy.signal
import statsmodels.formula.api as smf
import statsmodels.stats.anova
import mne
import mne_connectivity
from tqdm import tqdm
```

### Hardware Requirements
- Sufficient RAM for EEG data processing (>8GB recommended)
- Storage space for large physiological datasets

## Usage

1. **Data Preparation**: Ensure all data files are in the expected directory structure
2. **Individual Analysis**: Run each comparison script independently:
   ```bash
   python compare_controller_action.py
   python compare_eeg_TI.py
   python compare_eye_data.py
   python compare_performance.py
   python compare_questionnaire.py
   python compare_speech_event.py
   ```
3. **Output**: Generated plots will be saved in the `plots/` directory

## Key Findings Areas

The analysis framework is designed to investigate:

- **Behavioral Coordination**: How action patterns differ between human-only and human-AI teams
- **Neural Synchronization**: Whether human-AI teams show different brain coupling patterns
- **Cognitive Load**: Pupil-based indicators of mental effort and attention
- **Communication Patterns**: Speech frequency and duration differences
- **Performance Outcomes**: Task performance across different conditions
- **Subjective Experience**: Perceived helpfulness and leadership dynamics


## Citation
	@article{qin2025perception,
	  title={Perception of an AI Teammate in an Embodied Control Task Affects Team Performance, Reflected in Human Teammates' Behaviors and Physiological Responses},
	  author={Qin, Yinuo and Lee, Richard T and Sajda, Paul},
	  journal={arXiv preprint arXiv:2501.15332},
	  year={2025}
	}
