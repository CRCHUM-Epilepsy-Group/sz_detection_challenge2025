#!/usr/bin/env ipython

from rich.console import Console
import os
import polars as pl
from bids.layout import parse_file_entities
from epileptology.features.featureextraction import FeatureExtractor
import epileptology.preprocessing as pp
from szdetect import project_settings as s
from pathlib import Path

os.environ["INPUT"] = (
    # 395MB file
    "/mnt/data/SeizureDetectionChallenge2025/BIDS_Siena/sub-14/ses-01/eeg/sub-14_ses-01_task-szMonitoring_run-02_eeg.edf"
)


def main():
    console = Console()
    in_docker = os.environ.get("IN_DOCKER", False)
    if in_docker:
        edf_file = f"/data/{os.environ.get('INPUT')}"
    else:
        edf_file = f"{os.environ.get('INPUT')}"
    dataset_name = "test_set"

    eeg = pp.read_edf(edf_file, **s.PREPROCESSING_KWARGS["read_edf"])
    eeg = pp.filter_eeg(eeg, **s.PREPROCESSING_KWARGS["filter_eeg"])
    eeg = eeg.apply_function(pp.normalize_eeg)
    eeg = pp.segment_overlapping_windows(eeg, **s.PREPROCESSING_KWARGS["segment_eeg"])

    n_epochs = len(eeg.events)
    step_size = s.PREPROCESSING_KWARGS["segment_eeg"]["step_size"]

    console.log(
        f"EEG data loaded and preprocessed. Total number of epochs: {n_epochs:_}. Step size: {step_size}s"
    )
    batch_size = 200

    extractor = FeatureExtractor(
        s.FEATURES,
        s.FRAMEWORKS,
        log_dir=s.FEATURE_LOG_DIR,
        num_workers=s.NUM_WORKERS,
        console=console,
    )

    file_entities = parse_file_entities(edf_file)
    subject = file_entities["subject"]
    session = file_entities["session"]
    run = file_entities["run"]
    unique_id = f"{dataset_name}_{subject}_{session}_{run}"

    for i in range(0, n_epochs, batch_size):
        console.log(f"Computing features for epochs {i}:{i + batch_size}")
        batch_end = min(i + batch_size, n_epochs)
        epoch_batch = eeg[i:batch_end]

        parquet_sink = s.FEATURES_DIR / f"{unique_id}_batch-{i}.parquet"

        filename = Path(edf_file).name

        log_file = f"{filename}_batch-{i}"

        try:
            features = extractor.extract_feature(epoch_batch, log_file)

            features = features.with_columns(
                dataset_name=pl.lit(dataset_name),
                subject=pl.lit(subject),
                session=pl.lit(session),
                run=pl.lit(run),
                unique_id=pl.lit(unique_id),
                second=(pl.col("epoch").cast(pl.Int32) * step_size),
            )
            features.write_parquet(parquet_sink)
            console.log("Features saved to file")

        except (FileNotFoundError, IndexError, ValueError) as e:
            error_log = {"Patient_filename": str(edf_file), "Error": str(e)}
            print(error_log)
            continue


if __name__ == "__main__":
    main()
