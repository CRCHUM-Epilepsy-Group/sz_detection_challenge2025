#!/usr/bin/env ipython
import os
from pathlib import Path

import epileptology.preprocessing as pp
import tomllib
from epileptology.features.featureextraction import FeatureExtractor
from epileptology.utils.parsers import parse_config, parse_featureextraction_config

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# Overall settings
PROJECT = config["project"]["name"]

# Datasets
BIDS_DATASETS = {k: Path(v) for k, v in config["datasets"].items()}

# Bids DB file
BIDS_DB_FILES_DIR = Path(config["utilities"]["bids_db_files_dir"])

# Labels
LABELS_FILE = config["labels"]["labels_file"]

# Feature extraction
FEATURES, FRAMEWORKS = parse_featureextraction_config(
    config["features"]["features_config"]
)
FEATURES_DIR = Path(config["features"]["features_dir"])
NUM_WORKERS = config["features"]["num_workers"]
OVERWRITE_FEATURES = config["features"]["overwrite"]
FEATURE_LOG_DIR = config["features"]["log_dir"]

PREPROCESSING_KWARGS = config["preprocessing"]
# HACK to replace -1 with None (not allowed in TOML)
for step, kwargs in PREPROCESSING_KWARGS.items():
    for kwarg_name, value in kwargs.items():
        if value == -1:
            value = None
        PREPROCESSING_KWARGS[step][kwarg_name] = value

# Runtime env
DEBUG = config["runtime"]["debug"]
MAX_N_EEG = config["runtime"]["max_n_eeg"]
IN_DOCKER = os.environ.get("IN_DOCKER") == "1"

# Results
RESULTS_DIR = Path(config["results"]["results_dir"])

# Output of inference
OUTPUT_DIR = Path(config["output"]["output_dir"])

# Pretrained model
MODEL_FILE = Path(config["model"]["pretrained_model_file"])
SCALER_FILE = Path(config["model"]["pretrained_scaler_file"])
MRMR_FILE = Path(config["model"]["pretrained_mrmr_file"])

# Firing power
TAU = config["firing_power"]["tau"]
THRESHOLD = config["firing_power"]["threshold"]
# Pretrained pipeline and pipeline steps
PIPE_DIR = Path(config["pipe"]["pipe_dir"])
PIPE_DIR.mkdir(parents=True, exist_ok=True)
PIPE_FILE = Path(config["pipe"]["pipe_file"])
PIPE_FILE.parent.mkdir(parents=True, exist_ok=True)

# Logs
LOGS_FILE = Path(config["logs"]["log_file"])

if "PYTEST_CURRENT_TEST" not in os.environ:
    if not IN_DOCKER:
        BIDS_DB_FILES_DIR.mkdir(parents=True, exist_ok=True)
        Path(LABELS_FILE).parent.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOGS_FILE.parent.mkdir(parents=True, exist_ok=True)


FEATURE_GROUPS = {
    "efficiency": [
        "betweenness",
        "diversity_coef",
        "node_betweenness",
        "participation_coef",
        "module_degree_zscore",
        "eigenvector_centrality",
        "efficiency",
        "global_diffusion_efficiency",
        "global_rout_efficiency",
        "local_rout_efficiency",
    ],
    "connectivity": [
        "node_degree",
        "node_strength",
        "transitivity",
        "eigenvalues",
    ],
    "univariate": [
        "fuzzen",
        "linelength",
        "corr_dim",
        "band_power",
        "peak_alpha",
    ],
}
