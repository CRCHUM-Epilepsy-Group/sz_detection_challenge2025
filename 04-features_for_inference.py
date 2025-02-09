#!/usr/bin/env ipython

import os
import polars as pl
from epileptology.utils.toolkit import calculate_over_pool
from rich.console import Console
from bids.layout import parse_file_entities
from epileptology.features.featureextraction import FeatureExtractor
import epileptology.preprocessing as pp
from szdetect import project_settings as s
from pathlib import Path


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
    edf_file = f"{os.environ.get('INPUT')}"
    dataset_name = "test_set"

    eeg = pp.read_edf(edf_file, **s.PREPROCESSING_KWARGS["read_edf"])
    eeg = pp.filter_eeg(eeg, **s.PREPROCESSING_KWARGS["filter_eeg"])
    eeg = eeg.apply_function(pp.normalize_eeg)
    eeg = pp.segment_overlapping_windows(eeg, **s.PREPROCESSING_KWARGS["segment_eeg"])
    extractor = FeatureExtractor(
        s.FEATURES, s.FRAMEWORKS, log_dir=s.FEATURE_LOG_DIR, num_workers=num_workers
    )

    n_epochs = len(eeg)
    batch_size = 10
    extractor = FeatureExtractor(
        s.FEATURES,
        s.FRAMEWORKS,
        log_dir=s.FEATURE_LOG_DIR,
        num_workers=s.NUM_WORKERS,
    )

    file_entities = parse_file_entities(path)
    subject = file_entities["subject"]
    session = file_entities["session"]
    run = file_entities["run"]
    unique_id = f"{dataset_name}_{subject}_{session}_{run}"

    for i in range(0, n_epochs, batch_size):
        batch_end = min(i + batch_size, n_epochs)
        epoch_batch = eeg[i:batch_end]

        parquet_sink = s.FEATURES_DIR / f"{unique_id}_batch{i}.parquet"

        filename = Path(edf_file).name

        log_file = f"{filename}_batch{i}"

        features = extractor.extract_feature(epoch_batch, log_file)

        features = features.with_columns(
            dataset_name=pl.lit(dataset_name),
            subject=pl.lit(subject),
            session=pl.lit(session),
            run=pl.lit(run),
            unique_id=pl.lit(unique_id),
            second=pl.col("epoch").cast(pl.Int32),
        )
        features.write_parquet(parquet_sink)


if __name__ == "__main__":
    main()
