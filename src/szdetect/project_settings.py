#!/usr/bin/env ipython
import duckdb
from epileptology.features.featureextraction import FeatureExtractor
import epileptology.preprocessing as pp
from pathlib import Path
from epileptology.utils.parsers import parse_config, parse_featureextraction_config

config = parse_config("proj_config.yaml")

# Overall settings
PROJECT = config["project"]

# Datasets
TUH_SZ_BIDS = Path(config["datasets"]["tuh_sz_bids"])
CHB_MIT_BIDS = Path(config["datasets"]["chb_mit_bids"])
SIENA_BIDS = Path(config["datasets"]["siena_bids"])
BIDS_DATASETS = {k: Path(v) for k, v in config["datasets"].items()}

# Feature extraction
FEATURES_DB = config["features"]["features_db_file"]
FEATURES, FRAMEWORKS = parse_featureextraction_config(
    config["features"]["features_config"]
)
NUM_WORKERS = config["features"]["num_workers"]

PREPROCESSING_KWARGS = {
    "read_edf": {
        "channels": None,  # Default to the 19 channels in 10-20 system
    },
    "filter_eeg": {
        "l_freq": 1,
        "h_freq": 99,
        "notch_filter": None,
    },
    "resameple_eeg": {
        "sfreq_new": 200,
    },
    "segment_eeg": {
        "window_duration": 15,
        "overlap": 1,
    },
    # "artifact_correction": {
    #     "n_interpolates": np.array([1, 5, 9]),
    #     "n_consensus_percs": 4,
    #     "cv": 5,
    # },
}

# Runtime env
DEBUG = config["runtime"]["debug"]
