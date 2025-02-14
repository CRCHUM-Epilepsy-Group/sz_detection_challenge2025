#!/usr/bin/env ipython


from szdetect import model as mod
import polars as pl


def firing_power_pl(pred_series: pl.Series, tau, threshold):
    # out = df.group_by("keys").agg(pl.col("values").map_batches(fn))
    reg_pred = mod.firing_power(pred_series.to_numpy(), tau=tau)
    return pl.Series([1 if x >= threshold else 0 for x in reg_pred])


def get_events_df(df, step_size: int):
    # First, identify groups of consecutive 1s
    df = df.with_columns(
        # Mark start of new groups when fp_pred changes
        pl.col("fp_pred").diff().fill_null(0).abs().alias("group_start"),
        # Keep seconds for calculation
        pl.col("second").alias("second_orig"),
    )

    # Create group IDs for consecutive predictions
    df = df.with_columns(pl.col("group_start").cum_sum().alias("group_id"))

    # Filter for groups with fp_pred=1 and aggregate
    events = (
        df.filter(pl.col("fp_pred") == 1)
        .group_by(["dataset_name", "subject", "session", "run", "group_id"])
        .agg(
            onset=pl.col("second").min(),
            duration=pl.col("second").max()
            - pl.col("second").min()
            + step_size,  # assuming 4s epochs
            dateTime=pl.col("timestamp").first(),
        )
        .drop("group_id")
    )

    # Add eventType column
    events = events.with_columns(pl.lit("sz").alias("eventType"))

    # Handle cases with no events
    all_groups = df.select(["dataset_name", "subject", "session", "run"]).unique()
    events = all_groups.join(
        events, on=["dataset_name", "subject", "session", "run"], how="left"
    ).with_columns(
        pl.col("onset").cast(pl.Float64), pl.col("duration").cast(pl.Float64)
    )

    return events
