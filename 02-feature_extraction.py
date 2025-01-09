#!/usr/bin/env ipython
import time
import polars as pl
from epileptology.utils.toolkit import calculate_over_pool
from rich.console import Console
from bids.layout import parse_file_entities
from epileptology.features.featureextraction import FeatureExtractor
import epileptology.preprocessing as pp
from bids import BIDSLayout
from szdetect import project_settings as s


def feature_extraction_pipeline(
    iterated,
    features,
    frameworks,
    segmenting_function,
    preprocessing_kwargs,
):
    dataset_name, path = iterated
    extraction_start_time = time.perf_counter()
    try:
        eeg = pp.read_edf(path, **preprocessing_kwargs["read_edf"])
        # Normalize EEG
        eeg = pp.filter_eeg(eeg, **preprocessing_kwargs["filter_eeg"])
        eeg = eeg.apply_function(pp.normalize_eeg)
        eeg = segmenting_function(eeg, **preprocessing_kwargs["segment_eeg"])
        extractor = FeatureExtractor(features, frameworks)
        features = extractor.extract_feature(eeg)

        subject = parse_file_entities(path)["subject"]
        session = parse_file_entities(path)["session"]
        run = parse_file_entities(path)["run"]
        unique_id = f"{dataset_name}_{subject}_{session}_{run}"

        features = features.with_columns(
            dataset_name=pl.lit(dataset_name),
            subject=pl.lit(subject),
            session=pl.lit(session),
            run=pl.lit(run),
            unique_id=pl.lit(unique_id),
            second=pl.col("epoch").cast(pl.Int32),
        )
        print("Writing parquet to file")
        features.write_parquet(s.FEATURES_DIR / f"{unique_id}.parquet")
        extraction_duration = time.perf_counter() - extraction_start_time
        print(f"Features extracted for {unique_id} in {extraction_duration:.2f}s")

        return 1

    except (FileNotFoundError, IndexError, ValueError) as e:
        error_log = {"Patient_filename": str(path), "Error": str(e)}
        print(error_log)
        return 0


def main():
    console = Console()

    # TODO: add database_path to Extract_labels script to speed up parsing of BIDS dbs
    bids_datasets = {
        name: BIDSLayout(path, database_path=(s.BIDS_DB_FILES_DIR / f"{name}.db"))
        for name, path in s.BIDS_DATASETS.items()
    }

    # Get a list of all EEG files
    eeg_files = {
        name: bids.get(extension=".edf", return_type="filename")
        for name, bids in bids_datasets.items()
    }
    name_file_pairs = [(name, f) for name, files in eeg_files.items() for f in files]
    if s.DEBUG:
        # Sample from each dataset
        name_file_pairs = [
            (name, f) for name, files in eeg_files.items() for f in files[:2]
        ]

    features = calculate_over_pool(
        feature_extraction_pipeline,
        name_file_pairs,
        num_workers=s.NUM_WORKERS,
        debug=s.DEBUG,
        features=s.FEATURES,
        frameworks=s.FRAMEWORKS,
        segmenting_function=pp.segment_overlapping_windows,
        preprocessing_kwargs=s.PREPROCESSING_KWARGS,
        chunksize=4,
        console=console,
        n_jobs=len(name_file_pairs),
    )


if __name__ == "__main__":
    main()
