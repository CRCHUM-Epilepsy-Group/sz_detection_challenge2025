#!/usr/bin/env ipython
import numpy as np
import polars as pl
import pytest

from szdetect.predictions_to_df import firing_power_pl, get_events_df


@pytest.fixture
def pred_dl():
    return pl.DataFrame(
        {
            "dataset_name": [
                "mock_1",
                "mock_1",
                "mock_1",
                "mock_1",
                "mock_1",
                "mock_1",
                "mock_1",
                "mock_1",
                "mock_2",
                "mock_2",
            ],
            "subject": ["01", "01", "01", "01", "01", "01", "01", "01", "01", "01"],
            "session": ["01", "01", "01", "01", "01", "01", "01", "01", "01", "02"],
            "run": ["01", "01", "01", "01", "01", "01", "01", "01", "01", "01"],
            "y_pred": [0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
            "second": [0, 4, 8, 12, 16, 20, 24, 28, 0, 0],
            "timestamp": [
                "2021-01-01 00:00:00",
                "2021-01-01 00:10:00",
                "2021-01-01 00:20:00",
                "2021-01-01 00:30:00",
                "2021-01-01 00:40:00",
                "2021-01-01 00:50:00",
                "2021-01-01 00:60:00",
                "2021-01-02 00:20:00",
                "2021-01-02 00:30:00",
                "2021-01-02 00:40:00",
            ],
        }
    )


@pytest.fixture
def pred_series(pred_dl):
    return pred_dl["y_pred"]


@pytest.fixture
def fp_pred_dl(pred_dl):
    tau = 2
    threshold = 0.5
    return pred_dl.with_columns(
        pl.col("y_pred")
        .over(["dataset_name", "subject", "session", "run"])
        .sort_by("second")
        .map_batches(lambda x: firing_power_pl(x, tau, threshold))
        .alias("fp_pred")
    )


def test_firing_power_pl(pred_series):
    fp = firing_power_pl(pred_series, 2, 0.5)

    assert fp.dtype == pl.Int64


def test_batch_apply_firing_power(pred_dl):
    tau = 2
    threshold = 0.5
    new_df = pred_dl.with_columns(
        pl.col("y_pred")
        .over(["dataset_name", "subject", "session", "run"])
        .sort_by("second")
        .map_batches(lambda x: firing_power_pl(x, tau, threshold))
        .alias("fp_pred")
    )

    assert True


def test_pred_to_events(fp_pred_dl):
    event_df = get_events_df(fp_pred_dl, 4)
    assert True
