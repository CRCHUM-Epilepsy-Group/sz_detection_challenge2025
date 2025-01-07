#!/usr/bin/env ipython
import polars as pl
from epileptology.utils.toolkit import calculate_over_pool
from rich.console import Console
from bids.layout import parse_file_entities
from epileptology.features.featureextraction import FeatureExtractor
import epileptology.preprocessing as pp
from bids import BIDSLayout
from szdetect import project_settings as s
import itertools


def feature_extraction_pipeline(
    iterated,
    features,
    frameworks,
    segmenting_function,
    preprocessing_kwargs,
):
    dataset_name, path = iterated
    try:
        eeg = pp.read_edf(path, **preprocessing_kwargs["read_edf"])
        # Normalize EEG
        eeg = pp.filter_eeg(eeg, **preprocessing_kwargs["filter_eeg"])
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
        ).rename({"time": "second"})
        features.write_parquet(s.FEATURES_DIR / f"{unique_id}.parquet")
        print(f"Features extracted for {unique_id}")

        return features

    except (FileNotFoundError, IndexError, ValueError) as e:
        error_log = {"Patient_filename": str(path), "Error": str(e)}
        print(error_log)
        return


def main():
    console = Console()

    bids_datasets = {
        name: BIDSLayout(path, database_path=(path / "bids.db"))
        for name, path in s.BIDS_DATASETS.items()
    }
    session_ids_by_dataset = {
        name: bids.get_sessions() for name, bids in bids_datasets.items()
    }

    if s.DEBUG:
        session_ids_by_dataset = {
            name: ids[:4] for name, ids in session_ids_by_dataset.items()
        }

    # Get a list of all EEG files
    eeg_files = itertools.chain(
        *[
            bids.get(session=eeg_list, extension=".edf", return_type="filename")
            for bids, eeg_list in zip(
                bids_datasets.values(), session_ids_by_dataset.values()
            )
        ]
    )
    dataset_names = itertools.chain(*[name for name in bids_datasets.keys()])

    features = calculate_over_pool(
        feature_extraction_pipeline,
        zip(dataset_names, eeg_files),
        num_workers=s.NUM_WORKERS,
        debug=s.DEBUG,
        features=s.FEATURES,
        frameworks=s.FRAMEWORKS,
        segmenting_function=pp.segment_overlapping_windows,
        preprocessing_kwargs=s.PREPROCESSING_KWARGS,
    )


if __name__ == "__main__":
    main()
