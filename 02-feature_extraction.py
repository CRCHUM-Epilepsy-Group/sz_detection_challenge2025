#!/usr/bin/env ipython
import duckdb
import polars as pl
from epileptology.utils.toolkit import calculate_over_pool
from rich.console import Console
from bids.layout import parse_file_entities
from epileptology.features.featureextraction import FeatureExtractor
import epileptology.preprocessing as pp
from bids import BIDSLayout
from szdetect import project_settings as s
from pathlib import Path
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
        eeg = pp.filter_eeg(eeg, **preprocessing_kwargs["filter_eeg"])
        eeg = segmenting_function(eeg, **preprocessing_kwargs["segment_eeg"])
        # cleaned_eeg = pp.clean_eeg(
        #     eeg_seg, **preprocessing_kwargs["artifact_correction"]
        # )
        extractor = FeatureExtractor(features, frameworks)
        features = extractor.extract_feature(eeg)

        session = parse_file_entities(path)["session"]
        dataset = parse_file_entities(path)

        features = features.with_columns(
            session=pl.lit(session),
            dataset=pl.lit(dataset_name),
        )

        return features

    except (FileNotFoundError, IndexError, ValueError) as e:
        error_log = {"Patient_filename": str(path), "Error": str(e)}
        print(error_log)
        return


def main():
    console = Console()

    bids_datasets = {
        name: BIDSLayout(path, path / "bids.db")
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

    df = pl.concat(features)
    with duckdb.connect(s.FEATURES_DB) as con:
        con.execute("""CREATE OR REPLACE TABLE features AS FROM df""")

    print("data transferred to db")


if __name__ == "__main__":
    main()
