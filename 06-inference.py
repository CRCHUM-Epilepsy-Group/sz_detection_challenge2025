import pickle
from szdetect.predictions_to_df import firing_power_pl, get_events_df
import os
import polars as pl
from pathlib import Path
from szdetect import pull_features as pf
from szdetect.write_predictions import write_predictions
from szdetect import project_settings as s


def main():
    df = pf.pull_features(
        feature_dir=s.FEATURES_DIR,
        feature_group="all",
        inference=True,
    )

    index_col = [
        "timestamp",
        "second",
        "dataset_name",
        "subject",
        "session",
        "run",
    ]

    feature_col = ["region_side", "freqs", "feature"]

    wide_df = df.select(index_col + feature_col + ["value"]).pivot(
        values="value", index=index_col, on=feature_col, maintain_order=True
    )
    print("long to wide pivot succeeded.")

    model = pickle.load(open(s.MODEL_FILE, "rb"))
    scaler = pickle.load(open(s.SCALER_FILE, "rb"))
    mrmr = pickle.load(open(s.MRMR_FILE, "rb"))

    X_test = wide_df.drop(index_col)
    X_test_fs = mrmr.transform(X_test.to_pandas())
    X_test_scaled = scaler.transform(X_test_fs)
    y_pred = model.predict(X_test_scaled)

    wide_df = wide_df.select(index_col).with_columns(pl.lit(y_pred).alias("y_pred"))

    # Firing power
    tau = s.TAU
    threshold = s.THRESHOLD
    df_fp = wide_df.with_columns(
        pl.col("y_pred")
        .over(["dataset_name", "subject", "session", "run"])
        .sort_by("second")
        .map_batches(lambda x: firing_power_pl(x, tau, threshold))
        .alias("fp_pred")
    )
    event_df = get_events_df(df_fp, s.PREPROCESSING_KWARGS["segment_eeg"]["step_size"])

    # Event predictions to TSV file with colomns onset, duration, evetType (sz) and dateTime
    # in POSIX foramt %Y-%m-%d %H:%M:%S

    if s.IN_DOCKER:
        output_file = Path(f"/output/{os.environ.get('OUTPUT')}")  # type: ignore
        OUTPUT_DIR = output_file.parent
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        OUTPUT_DIR = s.OUTPUT_DIR
        output_file = None

    # Clear output directory
    # for file in OUTPUT_DIR.glob("*"):
    #     file.unlink()

    write_predictions(event_df, OUTPUT_DIR, output_file=output_file)


if __name__ == "__main__":
    main()
