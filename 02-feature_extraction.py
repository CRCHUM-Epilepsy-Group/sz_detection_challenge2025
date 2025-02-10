#!/usr/bin/env ipython
import os

# For testing single EEG times
# os.environ["IN_DOCKER"] = "1"
# os.environ["INPUT"] = (
#     "/mnt/data/SeizureDetectionChallenge2025/BIDS_tuh_eeg_seizure/sub-386/ses-04/eeg/sub-386_ses-04_task-szMonitoring_run-00_eeg.edf"
# )
import time
import polars as pl
from epileptology.utils.toolkit import calculate_over_pool
from rich.console import Console
from bids.layout import parse_file_entities
from epileptology.features.featureextraction import FeatureExtractor
import epileptology.preprocessing as pp
from bids import BIDSLayout
from szdetect import project_settings as s
from pathlib import Path
import random


def feature_extraction_pipeline(
    iterated,
    features,
    frameworks,
    segmenting_function,
    preprocessing_kwargs,
    num_workers=1,
):
    dataset_name, path = iterated

    file_entities = parse_file_entities(path)
    subject = file_entities["subject"]
    session = file_entities["session"]
    run = file_entities["run"]
    unique_id = f"{dataset_name}_{subject}_{session}_{run}"
    parquet_sink = s.FEATURES_DIR / f"{unique_id}.parquet"

    filename = Path(path).name

    if parquet_sink.exists() and not s.OVERWRITE_FEATURES:
        print(f"Features already extracted for {filename}")

    try:
        eeg = pp.read_edf(path, **preprocessing_kwargs["read_edf"])
        # Normalize EEG
        eeg = pp.filter_eeg(eeg, **preprocessing_kwargs["filter_eeg"])
        eeg = eeg.apply_function(pp.normalize_eeg)
        eeg = segmenting_function(eeg, **preprocessing_kwargs["segment_eeg"])
        extractor = FeatureExtractor(
            features, frameworks, log_dir=s.FEATURE_LOG_DIR, num_workers=num_workers
        )
        features = extractor.extract_feature(eeg, filename)

        features = features.with_columns(
            dataset_name=pl.lit(dataset_name),
            subject=pl.lit(subject),
            session=pl.lit(session),
            run=pl.lit(run),
            unique_id=pl.lit(unique_id),
            second=pl.col("epoch").cast(pl.Int32),
        )
        features.write_parquet(parquet_sink)

        return 1

    except (FileNotFoundError, IndexError, ValueError) as e:
        error_log = {"Patient_filename": str(path), "Error": str(e)}
        print(error_log)
        return 0


def main():
    console = Console()

    if s.IN_DOCKER:
        edf_file = f"{os.environ.get('INPUT')}"
        dataset_name = "testing_set"
        name_file_pair = (dataset_name, edf_file)

        feature_extraction_pipeline(
            name_file_pair,
            features=s.FEATURES,
            frameworks=s.FRAMEWORKS,
            segmenting_function=pp.segment_overlapping_windows,
            preprocessing_kwargs=s.PREPROCESSING_KWARGS,
            num_workers=s.NUM_WORKERS,
        )

    else:
        bids_datasets = {
            name: BIDSLayout(path, database_path=(s.BIDS_DB_FILES_DIR / f"{name}.db"))
            for name, path in s.BIDS_DATASETS.items()
        }

        # Get a list of all EEG files
        eeg_files = {
            name: bids.get(extension=".edf", return_type="filename")
            for name, bids in bids_datasets.items()
        }
        name_file_pairs = [
            (name, f) for name, files in eeg_files.items() for f in files
        ]

        if s.DEBUG:
            # Sample from each dataset
            name_file_pairs = [
                (name, f) for name, files in eeg_files.items() for f in files[:2]
            ]
        if s.MAX_N_EEG > 0:
            random.seed(123)
            name_file_pairs = name_file_pairs[: s.MAX_N_EEG - 1]
            random.shuffle(name_file_pairs)

        # _ = calculate_over_pool(
        #     feature_extraction_pipeline,
        #     name_file_pairs,
        #     num_workers=s.NUM_WORKERS,
        #     debug=s.DEBUG,
        #     features=s.FEATURES,
        #     frameworks=s.FRAMEWORKS,
        #     segmenting_function=pp.segment_overlapping_windows,
        #     preprocessing_kwargs=s.PREPROCESSING_KWARGS,
        #     chunksize=4,
        #     n_jobs=len(name_file_pairs),
        #     console=console,
        # )

        for name_file_pair in name_file_pairs:
            feature_extraction_pipeline(
                name_file_pair,
                features=s.FEATURES,
                frameworks=s.FRAMEWORKS,
                segmenting_function=pp.segment_overlapping_windows,
                preprocessing_kwargs=s.PREPROCESSING_KWARGS,
                num_workers=s.NUM_WORKERS,
            )


if __name__ == "__main__":
    main()
