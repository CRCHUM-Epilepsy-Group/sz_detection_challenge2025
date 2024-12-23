import sys
import polars as pl
import re, tomllib
import importlib.util

file_path = './01-pull_data.py'
spec = importlib.util.spec_from_file_location("pull_data", file_path)
pull_data = importlib.util.module_from_spec(spec)
sys.modules["pull_data"] = pull_data
spec.loader.exec_module(pull_data)


def get_events_info(tsv_file:str):
    """
    Retrieve info about EEG recording file from a _events.tsv file.
    Parameters:
        tsv_file (str): name of the _events.tsv file.
    Returns:
        dict: dataset_name (str), subject (str), session (str), run (str)
    """

    tsv_file = tsv_file.replace("\\", "/")
    pattern = r'^(?P<dataset_name>.+?/BIDS_[^/]+)/sub-(?P<subject>\d{2,3})/ses-(?P<session>\d{2})/eeg/sub-(?P=subject)_ses-(?P=session)_task-[^_]+_run-(?P<run>\d{2})_events\.tsv'
    match = re.search(pattern, tsv_file)

    if match:
        # Extract values
        dataset_name = match.group('dataset_name')
        subject = match.group('subject')
        session = match.group('session')
        run = match.group('run')
        
        return {'dataset_name': dataset_name,
                'subject': subject, 'session': session,'run': run}
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
    events_df = pl.read_csv(tsv_file, separator='\t')
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
        onset = event[0]
        duration = event[1]
        offset = onset + duration
    
        df = df.with_columns(
            pl.when((df['timestamp'] >= onset) & (df['timestamp'] <= offset))
            .then(True)
            .otherwise(df['seizure_label'])
            .alias('seizure_label')
        )
    return df

def generate_labeled_df(tsv_file):
    info = get_events_info(tsv_file)
    dataset_name = info['dataset_name']
    subject = info['subject']
    session = info['session']
    run = info['run']
    seizure_label = False
    
    sz_events = extract_sz_events(tsv_file)
    if sz_events.shape[0]:
        recording_duration = int(sz_events['recordingDuration'][0])
    else:
        events = pl.read_csv(tsv_file, separator='\t')
        recording_duration = int(events['recordingDuration'][0])

    timestamps = list(range(1, recording_duration + 1))
    
    df = pl.DataFrame({
    "dataset_name": [dataset_name] * recording_duration,
    "subject": [subject] * recording_duration,
    "session": [session] * recording_duration,
    "run": [run] * recording_duration,
    "timestamp": timestamps,
    "seizure_label": [seizure_label] * recording_duration})

    if sz_events.shape[0]:
        lb_df = update_seizure_labels(df, sz_events)
        return lb_df
    else: 
        return df


def main():
    # Load BIDS dataset
    with open(".\config.toml", "rb") as f:
        config = tomllib.load(f)

    bids_datasets = ['siena_bids','tuh_sz_bids', 'chb_mit_bids']

    events_tsv_files = []
    for db in bids_datasets:
        bids_directory = config['datasets'][db] 

        db_tsv_files = pull_data.get_bids_file_paths(bids_dir=bids_directory,
                                                     extension='tsv',
                                                     data_type='events')
        events_tsv_files.extend(db_tsv_files)

    final_df = None

    for tsv_file in events_tsv_files:
        labeled_df = generate_labeled_df(tsv_file)

        if final_df is None:
            final_df = labeled_df
        else:
            final_df = pl.concat([final_df, labeled_df], how="vertical")

    # Save to a Parquet file
    output_dir = config['output']['out_dir'] 
    final_df.write_parquet(output_dir + '/output_dataset.parquet')
    
    print("Parquet file has been generated successfully.")

if __name__ == '__main__':
    main()
