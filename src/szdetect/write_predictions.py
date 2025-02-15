def write_predictions(df, output_dir, output_file=None):
    for row in df.iter_rows(named=True):
        subject = row["subject"]
        session = row["session"]
        run = row["run"]
        onset = row["onset"]
        duration = row["duration"]
        eventType = row["eventType"]
        dateTime = row["dateTime"]
        recordingDuration = row["recordingDuration"]

        # Save to TSV file
        if output_file is None:
            write_to = (
                output_dir
                / f"sub-{subject}_ses-{session}_task-szMonitoring_run-{run}_events.tsv"
            )
        else:
            write_to = output_file
        # If file does not exists, create it and write the header
        if not write_to.exists():
            with open(write_to, "w") as f:
                f.write(
                    "onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration\n"
                )
        # If file exists, append the new event
        if row["onset"] is not None and row["duration"] is not None:
            with open(write_to, "a") as f:
                f.write(
                    f"{onset}\t{duration}\t{eventType}\tn/a\tn/a\t{dateTime}\t{recordingDuration}\n"
                )
