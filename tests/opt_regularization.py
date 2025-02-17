import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from timescoring import scoring
from timescoring import visualization
from timescoring.annotations import Annotation
from szdetect.model import firing_power


# Set your search grid
tau_range = np.arange(1, 25, 1) 
thresh_range = np.arange(0.05, 1, 0.05)
folds = [0]

for fold in folds:
    df = pl.read_parquet(f"/mnt/data/SeizureDetectionChallenge2025/data/pretrained_pipe_steps/y_pred_test.parquet")
    results = []

    for tau in tau_range:
        for threshold in thresh_range:
            sensitivites = []
            precisions = []
            f1s = []
            fpRates = []

            datasets = df.select("dataset_name").unique().to_series().to_list()
            for dataset in datasets:
                    df_dt = df.filter(pl.col("dataset_name")==dataset)
                    records = df_dt.select("unique_id").unique().to_series().to_list()
                    for record in records:
                        df_record = df_dt.filter(pl.col("unique_id")==record)
                        df_record = df_record.sort("second")

                        # firing power
                        y_pred = df_record["y_pred"].to_numpy()
                        y_true = df_record["label"].to_numpy()

                        y_pred_fp = firing_power(y_pred, tau=tau)
                        y_pred_reg = np.array([1 if x >= threshold else 0 for x in y_pred_fp])

                        # create ref and hyp
                        ref = Annotation(y_true, fs=1/4)
                        hyp = Annotation(y_pred_reg, fs=1/4)

                        param = scoring.EventScoring.Parameters(
                                toleranceStart=30,
                                toleranceEnd=30,
                                minOverlap=0,
                                maxEventDuration=5 * 60,
                                minDurationBetweenEvents=60)
                        
                        scores = scoring.EventScoring(ref, hyp, param)

                        sensitivites.append(scores.sensitivity)
                        precisions.append(scores.precision)
                        f1s.append(scores.f1)
                        fpRates.append(scores.fpRate)

            metrics_dict = {"tau": tau,
                                "threshold": threshold,
                                "sensitivity": np.nanmean(sensitivites),
                                "precision": np.nanmean(precisions),
                                "f1": np.nanmean(f1s),
                                "fpRate": np.nanmean(fpRates),
                                "f1_nan_counts": np.isnan(f1s).sum(),
                                "sensitivity_nan_counts": np.isnan(sensitivites).sum(),
                                "precision_nan_counts": np.isnan(precisions).sum()}
            
            results.append(metrics_dict) 

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"./Fold{fold}_results.csv", index=False)

    pivot_f1 = results_df.pivot(index="tau", columns="threshold", values="f1")
    pivot_precision = results_df.pivot(index="tau", columns="threshold", values="precision")
    pivot_sensitivity = results_df.pivot(index="tau", columns="threshold", values="sensitivity")

    # Create a single figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(32
                                            , 10))  # 1 row, 3 columns

    # Plot heatmaps
    sns.heatmap(pivot_f1, cmap="Blues", annot=True, fmt=".2f", linewidths=0.5, ax=axes[0])
    axes[0].set_title(f"Test_F1 Score")

    sns.heatmap(pivot_precision, cmap="Greens", annot=True, fmt=".2f", linewidths=0.5, ax=axes[1])
    axes[1].set_title(f"Test_Precision")

    sns.heatmap(pivot_sensitivity, cmap="Reds", annot=True, fmt=".2f", linewidths=0.5, ax=axes[2])
    axes[2].set_title(f"Test_Sensitivity")

    # Common labels
    for ax in axes:
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Tau")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"./Fold{fold}_heatmaps.png")
# calculate epoch-based metrics


"""
labels = pl.read_parquet("/mnt/data/SeizureDetectionChallenge2025/data/cleaned/labels.parquet")
labels_filtered = labels.filter(labels["label"] == True)
new_segment = labels_filtered["second"].diff().fill_null(0) < 0 
col_name = "second"  # Change this to the correct column name

# Step 1: Detect when a new range starts (when the column value decreases)
labels_filtered = labels_filtered.with_columns(
    new_segment=(labels_filtered[col_name].diff().fill_null(0) < 0).cast(pl.Int32)
)

# Step 2: Assign a group ID using `.cum_sum()`
labels_filtered = labels_filtered.with_columns(group_id=labels_filtered["new_segment"].cum_sum())

result_df = (
    labels_filtered.group_by(["unique_id", "dataset_name", "group_id"])
    .agg(pl.len().alias("duration"))
)

# Plot histogram
counts, bins, patches = plt.hist(durations, bins=30, edgecolor='black')

# Add the counts per bin and display bin range on top of each bar
for count, bin_left, patch in zip(counts, bins, patches):
    bin_right = bin_left + patch.get_width()
    # Label each bar with the count, positioning it above the bar
    plt.text(bin_left + patch.get_width() / 2, count + 1, f'{int(count)}', ha='center', va='bottom')
    # Display the range of the bin horizontally
    plt.text(bin_left + patch.get_width() / 2, -10, f'{int(bin_left)} - {int(bin_right)}', ha='center', va='top', rotation=270)

plt.title("Duration Distribution (0-200)")
plt.xlabel("Duration")
plt.ylabel("Frequency")
plt.show()
"""