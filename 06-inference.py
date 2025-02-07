import pickle
import os
import polars as pl
from pathlib import Path
from szdetect import pull_features as pf
from szdetect.write_predictions import write_predictions
from szdetect import project_settings as s


def main():
    df = pf.pull_features(
        feature_dir=s.FEATURES_DIR,
        label_file=s.LABELS_FILE,
        feature_group="all",
        inference=True,
    )

    index_col = ["dataset_name", "subject", "session", "run", "timestamp"]

    feature_col = ["region_side", "freqs", "feature"]

    wide_df = df.select(index_col + feature_col + ["value"]).pivot(
        values="value", index=index_col, on=feature_col, maintain_order=True
    )
    print("long to wide pivot succeeded.")

    # TODO add filepath to config and project_settings and load (s.MODEL_FILE)
    # model_file = s.MODEL_FILE
    # pretrained_model = pickle.load(open(model_file, "rb"))

    # X_test = wide_df.drop(index_col)
    # # X_test = scaler.transform(X_test_rec) # TODO add scaling on training data
    # y_pred = pretrained_model.predict(X_test)

    # Firing power

    # Event predictions to TSV file with colomns onset, duration, evetType (sz) and dateTime
    # in POSIX foramt %Y-%m-%d %H:%M:%S
    mock_predictions = pl.DataFrame(
        {
            "dataset_name": ["mock_1", "mock_1", "mock_1", "mock_2"],
            "subject": ["01", "01", "01", "01"],
            "session": ["01", "01", "02", "01"],
            "run": ["01", "02", "01", "01"],
            "onset": [0, 10, 20, 10],
            "duration": [10, 20, 20, 10],
            "eventType": ["sz", "sz", "sz", "sz"],
            "dateTime": [
                "2021-01-01 00:00:00",
                "2021-01-01 00:10:00",
                "2021-01-01 00:20:00",
                "2021-01-02 00:20:00",
            ],
        }
    )

    if s.IN_DOCKER:
        output_file = Path(f"/output/{os.environ.get('OUTPUT')}")  # type: ignore
        OUTPUT_DIR = output_file.parent
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        OUTPUT_DIR = s.OUTPUT_DIR
        output_file = None

    # Clear output directory
    for file in OUTPUT_DIR.glob("*"):
        file.unlink()

    write_predictions(mock_predictions, OUTPUT_DIR, output_file=output_file)


if __name__ == "__main__":
    main()
