# Seizure Detection Challenge: AI in Epilepsy 2025

[Instructions for the challenge](https://epilepsybenchmarks.com/challenge/)

# Workflow

1. Installation
* I recommend using uv (<https://docs.astral.sh/uv/getting-started/installation/>)

``` sh
wget -qO- https://astral.sh/uv/install.sh | sh
uv sync --frozen
```
* Install Git LFS (for test_data)
* Install epileptology

``` sh
# cd to a dedicated local directory, e.g.:
cd ../epileptology
git clone https://github.com/CRCHUM-Epilepsy-Group/epileptology/tree/main .
uv build
# cd back to your local project directory, e.g.:
cd ../sz_detection_challenge2025
# Install the epileptology package using uv
 UV_FIND_LINKS=/path/to/epileptology/dist
 # In my case: UV_FIND_LINKS=/Users/emilelemoine/software/epileptology/dist
 uv sync
```


2. Packages
* polars for DataFrames
* DuckDB for storage of features
* MNE-Python for EDF handling and EEG preprocessing
* PyBIDS for managing BIDS databases
* Zarr Arrays to store arrays on-disk
* See uv.lock or project.toml for requirements

3. Git worklow
* Start a branch to work on your feature
* When the code is ready: push the branch and go on GitHub to open a pull request

# Resources

[Review of winning features from previous competitions](https://github.com/Eldave93/Seizure-Detection-Tutorials/blob/master/02.%20Pre-Processing%20%26%20Feature%20Engineering.ipynb)

# Plan

## Initialize repo

```
- src \
  - chum_detection \ # Functions that will be imported by scripts
    - ... detection
    - ... feature extraction
    - features/
      - connectivity.py
      - ...
- 01-pull_data.py # Scripts in sequence
- 02-extract_features.py
...
- 05-classification.py
- ...
- main.py # The final script to be called when running the container
- tests \
  - test_data \
    - eeg_array.npy : (25 segments, 19 channels, 2000 timepoints)
    - conn_matrix.npy
    - features.db
  - test_connectivity.py
- explore \ # Files for exploration (notebooks, scripts, ...)
  - eigenvalue_corr.py
  - new_connectivty_features.py
```

## Data handling: BIDS / Zarr Arrays
OG and FH

* Goal: BIDS datasets stored locally
* Keep some testing data in reserve for final calibration steps

## Feature extraction
OG, IS, DG, and EL

* Testing data : EL
  * Get a toy DF with features
    * DuckDB for storage
    * Feature name, Electrode, Region, Wavelet level/freq band, EEG session, patient ID, timestamp, segment duration, value of feature, preprocessing vs. not preprocessed
  * Correlation matrix
  * EEG arrays
* Implement correlation conn and recurrence pre-transforms : EL
* Implement new features
  * [X] Entropy
  * [ ] Connectivity : IS and OG
    * [ ] Eigenvalue of correlation matrix
    * [ ] Covariance matrix
    * [ ] Recurrence plot
  * [X] PSD
  * [ ] New features : OG, IS, DG
* Start working on featureextraction pipeline: EL

## Seizure detection
* Start working on seizure detection pipeline: OG
