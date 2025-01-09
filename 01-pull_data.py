import mne
import tomllib
import polars as pl
from szdetect import pull_data


def read_edf_header(file_path):
    """
    Reads an EDF file using MNE and extracts header information.
    Parameters:
        file_path (str): Path to the EDF file.
    Returns:
        dict: A dictionary containing header and signal information.
    """
    try:
        # Load the EDF file header
        info, edf_info, _ = mne.io.edf.edf._get_info(file_path,
                                                    stim_channel='auto',
                                                    eog=None, misc=None,
                                                    exclude=(), infer_types=True,
                                                    preload=False)

        # Extract general header info
        header_info = {
            "File Name": file_path,
            "Measurement Date": info['meas_date'],
            "Number of Channels": len(info.ch_names),
            "Channel Names": info.ch_names,
            "Sampling Frequency (Hz)": info['sfreq'],
            "Duration (s)": edf_info['n_records'],
        }
        return header_info

    except Exception as e:
        print(f"Error reading EDF file: {e}")
        return None

# Function to load EEG data from .edf file
def load_eeg_data(edf_file):
    try:
        # Load the EEG data from the .edf file using MNE
        raw_data = mne.io.read_raw_edf(edf_file, preload=True)
        return raw_data
    except Exception as e:
        print(f"Error loading EEG data from {edf_file}: {e}")
        return None

# Function to load annotations from .tsv file
def load_annotations(tsv_file):
    """
    Read event annotations from a tsv file.

    Parameters:
        tsv_file (str): tsv file containing events.
    Returns:
        list: List of file paths matching the specified criteria.
    """
    try:
        # Load the annotations using polars
        annotations = pl.read_csv(tsv_file, separator='\t')
        return annotations
    except Exception as e:
        print(f"Error loading annotations from {tsv_file}: {e}")
        return None



def main():
    # Load BIDS dataset
    with open(".\config.toml", "rb") as f:
        config = tomllib.load(f)

    
    # bids datasets : tuh_sz_bids or chb_mit_bids or siena_bids
    bids_directory = config['datasets']['siena_bids'] 
    
    sub = '03'
    sess = '01'
    run = '00'
    
    # Load EEG files (.edf)
    edf_files = pull_data.get_bids_file_paths(bids_dir=bids_directory, extension='edf', 
                                subject=sub, session=sess, run=run,
                                data_type='eeg')

    # Load annotations files (.tsv)
    # tsv_files = get_bids_file_paths(bids_dir=bids_directory, extension='tsv', 
    #                           subject=sub, session=sess, run=run,
    #                           data_type='events')
    # This actually takes too much time
    tsv_files = [edf_file[:-7]+'events.tsv' for edf_file in edf_files]

    # Example: Load and process the first EEG file and annotation file
    if edf_files and tsv_files:
        eeg_data = load_eeg_data(edf_files[0])
        eeg_header = read_edf_header(edf_files[0])
        annotations = load_annotations(tsv_files[0])
        
        print(eeg_data.info)
        print(eeg_header)
        print(annotations.head())
        
    else:
        print("No EEG or annotation files found.")

if __name__ == '__main__':
    main()