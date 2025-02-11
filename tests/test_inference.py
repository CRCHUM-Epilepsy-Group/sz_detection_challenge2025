import pickle
import polars as pl
import numpy as np

from timescoring import scoring
from timescoring import visualization
from timescoring.annotations import Annotation

from szdetect import pull_features as pf
# from szdetect import project_settings as s
# from szdetect import model as mod


def grouper(iterable, thres=10):
    """ Group elements of an iterable into consecutive subgroups based on a threshold difference"""
    prev = None
    group = []
    for item in iterable:
        if prev is None or item - prev <= thres:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group
def firing_power(pre_input, tau=6):
    FPow = []
    try:
        assert tau > 1
        if len(pre_input) > tau:
            FPow = list(np.zeros(tau-1))
            for i in range(tau, len(pre_input)+1, 1):
                FP_n = np.sum(pre_input[i - tau:i]) / tau
                FPow.append(FP_n)
        else:
            FPow = pre_input
    except AssertionError:
        if tau == 1:
            FPow = pre_input
    return FPow


df = pf.pull_features(
    feature_dir="./test_data/features_v4",
    label_file="/mnt/data/SeizureDetectionChallenge2025/output/labels.parquet",
    # feature_dir=s.FEATURES_DIR,
    # label_file=s.LABELS_FILE,
    feature_group="all",
    inference=True,
    num_eegs=10
)

index_col = [
    "dataset_name",
    "subject",
    "session",
    "run",
    "unique_id",
    "timestamp",
    "second"
]

feature_col = ["region_side", "freqs", "feature"]

wide_df = df.select(index_col + feature_col + ["value"]).pivot(
    values="value", index=index_col, on=feature_col, maintain_order=True
)

print("long to wide pivot succeeded.")

# pipeline_file = s.MODEL_FILE
pipeline_file = "/mnt/data/SeizureDetectionChallenge2025/data/results/ml_run_1/fold_0_pipeline_3_7.sav"
pretrained_pipeline = pickle.load(open(pipeline_file, "rb"))

# step = s.PREPROCESSING_KWARGS['segment_eeg']['step_size']

# TODO to be adjusted
# tolerance = int(20 / step)
tolerance = 5 
tau = 3
threshold = 0.65


# TODO : optimize tau and threshold here 
datasets = wide_df.select("dataset_name").unique().to_series().to_list()
for dataset in datasets:
# for dataset in ['tuh_sz_bids']:
    df_dt = wide_df.filter(pl.col("dataset_name")==dataset)
    records = df_dt.select("unique_id").unique().to_series().to_list()
    # subjects = df_dt.select("subject").unique()
    # # for sub in subjects:
    # for sub in ['107']:
    #     df_dt_sub = df_dt.filter(pl.col("subject")==sub)
    #     sessions = df_dt_sub.select("session").unique()
    #     # for sess in sessions:
    #     for sess in ['04']:
    #         df_dt_sub_sess = df_dt_sub.filter(pl.col("session")==sess)
    #         runs = df_dt_sub_sess.select("run").unique()
    #         # for run in runs:
    #         for run in [2]:
    #             df_record = df_dt_sub_sess.filter(pl.col("run")==run)
    for record in records:
        print(record)
        df_record = df_dt.filter(pl.col("unique_id")==record)
        X = df_record.drop(index_col)
        
        y_pred = pretrained_pipeline.predict(X.to_pandas())
        # firing power
        y_pred = np.array(y_pred, dtype=int)
        y_pred_fp = firing_power(y_pred, tau=tau)
        y_pred_reg = np.array([1 if x >= threshold else 0 for x in y_pred_fp])

        # store if df 
        """ 
        pred_indexes = np.array([i for i, x in enumerate(y_pred_reg) if x == 1])
        pred_events = dict(enumerate(grouper(pred_indexes, thres=tolerance), 1))
        
        df_pred = df_record.select(index_col).with_row_index()
        df_pred = df_pred.with_columns(pl.Series("y_pred", y_pred_reg))

        if pred_events:
            print(f'Nb of predicted events {len(pred_events)}')
            for ev in pred_events:
                onset_idx = int(pred_events[ev][0])
                offset_idx = int(pred_events[ev][-1])
                onset = df_pred[onset_idx]["timestamp"][0]
                offset = df_pred[offset_idx]["timestamp"][0]
                dt = offset - onset
                duration = dt.total_seconds()
        """

                    



    

# Firing power
# Event predictions to TSV file with colomns onset, duration, evetType (sz) and dateTime
# in POSIX foramt %Y-%m-%d %H:%M:%S

