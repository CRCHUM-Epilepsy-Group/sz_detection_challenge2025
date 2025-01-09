#!/usr/bin/env ipython
import duckdb
from epileptology.features.featureextraction import FeatureExtractor
import epileptology.preprocessing as pp
from pathlib import Path
from epileptology.utils.parsers import parse_config, parse_featureextraction_config
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# Overall settings
PROJECT = config["project"]["name"]

# Datasets
TUH_SZ_BIDS = Path(config["datasets"]["tuh_sz_bids"])
CHB_MIT_BIDS = Path(config["datasets"]["chb_mit_bids"])
SIENA_BIDS = Path(config["datasets"]["siena_bids"])
BIDS_DATASETS = {k: Path(v) for k, v in config["datasets"].items()}

# Bids DB file
BIDS_DB_FILES_DIR = Path(config["utilities"]["bids_db_files_dir"])
BIDS_DB_FILES_DIR.mkdir(parents=True, exist_ok=True)

# Labels
LABELS_FILE = Path(config["labels"]["labels_file"])
LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Feature extraction
FEATURES_DB = config["features"]["features_db_file"]
FEATURES, FRAMEWORKS = parse_featureextraction_config(
    config["features"]["features_config"]
)
FEATURES_DIR = Path(config["features"]["features_dir"])
FEATURES_DIR.parent.mkdir(parents=True, exist_ok=True)
NUM_WORKERS = config["features"]["num_workers"]

PREPROCESSING_KWARGS = config["preprocessing"]
# HACK to replace -1 with None (not allowed in TOML)
for step, kwargs in PREPROCESSING_KWARGS.items():
    for kwarg_name, value in kwargs.items():
        if value == -1:
            value = None
        PREPROCESSING_KWARGS[step][kwarg_name] = value

# Runtime env
DEBUG = config["runtime"]["debug"]
