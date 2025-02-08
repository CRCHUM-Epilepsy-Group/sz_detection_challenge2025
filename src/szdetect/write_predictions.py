def write_predictions(df, output_dir, output_file=None):
    for row in df.iter_rows(named=True):
        subject = row["subject"]
        session = row["session"]
        run = row["run"]
        onset = row["onset"]
        duration = row["duration"]
        eventType = row["eventType"]
        dateTime = row["dateTime"]

        # Save to TSV file
        if output_file is None:
            output_file = (
                output_dir
                / f"sub-{subject}_ses-{session}_task-szMonitoring_run-{run}_events.tsv"
            )
        # If file does not exists, create it and write the header
        if not output_file.exists():
            with open(output_file, "w") as f:
                f.write("onset\tduration\teventType\tdatetime\n")
        # If file exists, append the new event
        with open(output_file, "a") as f:
            f.write(f"{onset}\t{duration}\t{eventType}\t{dateTime}\n")
