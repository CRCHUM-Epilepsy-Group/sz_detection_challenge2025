import sys
import random
import polars as pl
import re, tomllib
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from szdetect import project_settings as s

file_path = Path("./01-pull_data.py").resolve()
spec = importlib.util.spec_from_file_location("pull_data", file_path)
pull_data = importlib.util.module_from_spec(spec)
sys.modules["pull_data"] = pull_data
spec.loader.exec_module(pull_data)


def get_events_info(tsv_file: str):
    """
        Retrieve info about EEG recording file from a _events.tsv file.
    Parameters:
        tsv_file (str): name of the _events.tsv file.
    Returns:
        dict: dataset_name (str), subject (str), session (str), run (str)
    """
    tsv_file = tsv_file.replace("\\", "/")
    pattern = r"^(?P<dataset_name>.+?/BIDS_[^/]+)/sub-(?P<subject>\d{2,3})/ses-(?P<session>\d{2})/eeg/sub-(?P=subject)_ses-(?P=session)_task-[^_]+_run-(?P<run>\d{2})_events\.tsv"
    match = re.search(pattern, tsv_file)

    if match:
        # Extract values
        dataset_name = match.group("dataset_name")
        subject = match.group("subject")
        session = match.group("session")
        run = match.group("run")

        return {
            "dataset_name": dataset_name,
            "subject": subject,
            "session": session,
            "run": run,
        }
    else:
        print("No match found.")
        return None


def extract_sz_events(tsv_file):
    """
        Extracts events from a given TSV file using Polars.
    Parameters:
        tsv_file (str): Path to the events.tsv file.
    Returns:
        pl.DataFrame: A Polars DataFrame containing only seizure events, where eventType contains 'sz'.
    """
    events_df = pl.read_csv(tsv_file, separator="\t")
    sz_events_df = events_df.filter(events_df["eventType"].str.contains("sz"))
    return sz_events_df


def update_seizure_labels(df, sz_events_df):
    """
        Updates the seizure_label column to True for rows where timestamp falls between
        onset and (onset + duration) of seizure events.
    Parameters:
        df (pl.DataFrame): DataFrame containing the dataset with timestamp and seizure_label.
        sz_events_df (pl.DataFrame): DataFrame containing seizure event details with 'onset' and 'duration'.
    Returns:
        pl.DataFrame: Updated DataFrame with updated seizure labels.
    """
    for event in sz_events_df.iter_rows():
        onset = int(event[sz_events_df.columns.index("onset")])
        duration = int(event[sz_events_df.columns.index("duration")])
        offset = onset + duration

        df = df.with_columns(
            pl.when((df["second"] >= onset) & (df["second"] <= offset))
            .then(True)
            .otherwise(df["label"])
            .alias("label")
        )
    return df


def select_train_patients(patients: list, ratio=0.75):
    """
        Returns a list containing the ratio of the patients from the input patients list.
    Parameters:
        patients (list): The input list of all patients to sample from.
    Returns:
        train_patients: A list containing training patients.
    """
    if not patients:
        return []

    # TODO make sure testing set has 'enough' seizures
    train_patients = random.sample(patients, int(len(patients) * ratio))
    return train_patients


def update_train_label(df, patients):
    """
        Updates training label of dataframe to True for given patients.
    Parameters:
        df (pl.Dataframe): Dataframe including columns 'subject' and 'training'.
        patients (list): list of patients select for training.
    Returns:
        updated_df (pl.Dataframe): Dataframe with updated training column.
    """
    updated_df = df.with_columns((pl.col("subject").is_in(patients)).alias("training"))

    return updated_df


def generate_labeled_df(tsv_file, dataset):
    """
    Generate a polars dataframe from _events.tsv file.
    Parameters:
        tsv_file (str): path and name of the _events.tsv file.
        dataset (str): name of the dataset containing the tsv.
    Returns:
        pl.Dataframe: columns [subject, session, run, unique_id, dataset_name, second, timestamp, label, training]
    """
    info = get_events_info(tsv_file)
    # dataset_name = info['dataset_name']
    dataset_name = dataset
    # Here dataset is set according to provided name,
    # no verification is implemented at this level between provided dataset name and tsv file path
    # TODO add verification for dataset name
    """
    tsv_path = info['dataset_name']
    config_path = config['datasets'][dataset]
    assert Path(tsv_path).resolve() == Path(config_path).resolve(), 'Paths do not match'
    """

    subject = info["subject"]
    session = info["session"]
    run = info["run"]
    unique_id = f"{dataset}_{subject}_{session}_{run}"
    seizure_label = False

    sz_events = extract_sz_events(tsv_file)
    if sz_events.shape[0]:
        recording_duration = int(sz_events[0, "recordingDuration"])
        recording_dt = datetime.strptime(sz_events[0, "dateTime"], "%Y-%m-%d %H:%M:%S")
    else:
        events = pl.read_csv(tsv_file, separator="\t")
        recording_duration = int(events[0, "recordingDuration"])
        recording_dt = datetime.strptime(events[0, "dateTime"], "%Y-%m-%d %H:%M:%S")

    seconds = list(range(1, recording_duration + 1))
    timestamp = [
        recording_dt + timedelta(seconds=i) for i in range(1, recording_duration + 1)
    ]
    assert (
        len(timestamp) == recording_duration
    ), "Timestamp array length does not match recording duration."

    df = pl.DataFrame(
        {
            "unique_id": [unique_id] * recording_duration,
            "subject": [subject] * recording_duration,
            "session": [session] * recording_duration,
            "run": [run] * recording_duration,
            "dataset_name": [dataset_name] * recording_duration,
            "second": seconds,
            "timestamp": timestamp,
            "label": [seizure_label] * recording_duration,
            "training": [False] * recording_duration,
        }
    )

    if sz_events.shape[0]:
        lb_df = update_seizure_labels(df, sz_events)
        return lb_df
    else:
        return df


def main():
    final_df = None

    for db, bids_directory in s.BIDS_DATASETS.items():
        df = None

        db_tsv_files = pull_data.get_bids_file_paths(
            bids_dir=bids_directory, extension="tsv", data_type="events"
        )

        for tsv_file in db_tsv_files:
            labeled_df = generate_labeled_df(tsv_file, db)

            if df is None:
                df = labeled_df
            else:
                df = pl.concat([df, labeled_df], how="vertical")

        patients = df.select(pl.col("subject").unique()).to_series().to_list()
        train_patients = select_train_patients(patients)
        df = update_train_label(df, train_patients)

        if final_df is None:
            final_df = df
        else:
            final_df = pl.concat([final_df, df], how="vertical")

    # Save to a Parquet file
    final_df.write_parquet(s.LABELS_FILE)
    print("Parquet file has been generated successfully.")


if __name__ == "__main__":
    main()
