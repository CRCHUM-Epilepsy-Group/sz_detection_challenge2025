from bids import BIDSLayout

def get_bids_file_paths(bids_dir, extension, subject=None, session=None, run=None, data_type=None):
    """
    Retrieve file paths from a BIDS dataset based on specified filters.

    Parameters:
        bids_dir (str): Path to the BIDS dataset directory.
        extension (str): File extension to filter (e.g., 'edf').
        subject (str, optional): Subject identifier to filter (e.g., '01').
        session (str, optional): Session identifier to filter (e.g., '01').
        run (str, optional): Run identifier to filter (e.g., '01').
        data_type (str, optional): Data type to filter (e.g., 'eeg', 'meg').

    Returns:
        list: List of file paths matching the specified criteria.
    """
    # Initialize the BIDS layout
    layout = BIDSLayout(bids_dir)

    # Construct query with non-None parameters
    query = {
        "extension": extension,
        "subject": subject,
        "session": session,
        "run": run,
        "suffix": data_type
    }
    query = {key: value for key, value in query.items() if value is not None}
    
    files = layout.get(**query)
    
    return [file.path for file in files]