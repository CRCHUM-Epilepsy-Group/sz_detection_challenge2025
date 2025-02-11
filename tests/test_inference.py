import pickle
import polars as pl
import numpy as np
from szdetect import pull_features as pf
from szdetect import project_settings as s
from szdetect import model as mod


df = pf.pull_features(
    feature_dir=s.FEATURES_DIR,
    label_file=s.LABELS_FILE,
    feature_group="all",
    inference=True,
    num_eegs=10
)

index_col = [
    "timestamp",
    "dataset_name",
    "subject",
    "session",
    "run"
]

feature_col = ["region_side", "freqs", "feature"]

wide_df = df.select(index_col + feature_col + ["value"]).pivot(
    values="value", index=index_col, on=feature_col, maintain_order=True
)

print("long to wide pivot succeeded.")

pipeline_file = s.MODEL_FILE
pretrained_pipeline = pickle.load(open(pipeline_file, "rb"))

# step = s.PREPROCESSING_KWARGS['segment_eeg']['step_size']

# TODO to be adjusted
# tolerance = int(20 / step)
tolerance = 5 
tau = 3
threshold = 0.65

datasets = wide_df.select("dataset_name").unique()
# for dataset in datasets:
for dataset in ['tuh_sz_bids']:
    df_dt = wide_df.filter(pl.col("dataset_name")==dataset)
    subjects = df_dt.select("subject").unique()
    # for sub in subjects:
    for sub in ['107']:
        df_dt_sub = df_dt.filter(pl.col("subject")==sub)
        sessions = df_dt_sub.select("session").unique()
        # for sess in sessions:
        for sess in ['04']:
            df_dt_sub_sess = df_dt_sub.filter(pl.col("session")==sess)
            runs = df_dt_sub_sess.select("run").unique()
            # for run in runs:
            for run in [2]:
                df_record = df_dt_sub_sess.filter(pl.col("run")==run)
                X = df_record.drop(index_col)

                y_pred = pretrained_pipeline.predict(X.to_pandas())
                # firing power
                y_pred = np.array(y_pred, dtype=int)
                y_pred_fp = mod.firing_power(y_pred, tau=tau)
                y_pred_reg = np.array([1 if x >= threshold else 0 for x in y_pred_fp])

                pred_indexes = np.array([i for i, x in enumerate(y_pred_reg) if x == 1])
                pred_events = dict(enumerate(mod.grouper(pred_indexes, thres=tolerance), 1))

                df_pred = df_record.select(index_col).with_row_index()
                df_pred = df_pred.with_columns(pl.Series("y_pred", y_pred_reg))
                for ev in pred_events:
                    onset_idx = int(pred_events[ev][0])
                    offset_idx = int(pred_events[ev][-1])
                    onset = df_pred[onset_idx]["timestamp"][0]
                    offset = df_pred[offset_idx]["timestamp"][0]
                    



    

# Firing power
# Event predictions to TSV file with colomns onset, duration, evetType (sz) and dateTime
# in POSIX foramt %Y-%m-%d %H:%M:%S

