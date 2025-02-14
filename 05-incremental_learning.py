import glob
import pickle
import warnings
import polars as pl

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import xgboost as xgb

from feature_engine.selection import MRMR
from sklearn.preprocessing import StandardScaler
from szdetect import pull_features as pf
from szdetect import project_settings as s


def main():
    max_per_range = 2500
    files_count = len(glob.glob(str(s.FEATURES_DIR)+"/*.parquet"))
    print(f"files_count {files_count}")

    ranges = []
    start = 1

    while start <= files_count:
        end = min(start + max_per_range - 1, files_count)  # Ensure we don't exceed total_files
        ranges.append((start, end))
        start = end + 1  # Move to the next range

    # mid = files_count // 2 
    # ranges = [(i, min(i + mid - 1, files_count)) for i in range(0, files_count, mid)]
    # print(f"ranges {ranges}")

    params = {'max_depth': 7, 'min_child_weight': 15,
              'scale_pos_weight': 13, 
              'max_delta_step': 1,
              'eval_metric':'aucpr', 'reg_alpha': 100,
              'learning_rate': 0.05, 'gamma': 0.3, 'booster': 'gbtree'}

    sel = MRMR(method="FCQ", regression=False)
    sc = StandardScaler()
    model = xgb.XGBClassifier()
    model.set_params(**params)
        

    iteration = 0
    booster = None
    for r in ranges:
        print(f"iteration {iteration}")
        df = pf.pull_features(
            feature_dir=s.FEATURES_DIR,
            label_file=s.LABELS_FILE,
            feature_group="all",
            train_only=True,
            step_size=s.PREPROCESSING_KWARGS['segment_eeg']['step_size'],
            start_eeg=r[0],
            end_eeg=r[1],
        )

        index_col = [
            "dataset_name",
            "subject",
            "session",
            "run",
            "unique_id",
            "timestamp",
            "second",
            "label"
        ]

        feature_col = ["region_side", "freqs", "feature"]

        wide_df = df.select(index_col + feature_col + ["value"]).pivot(
            values="value", index=index_col, on=feature_col, maintain_order=True
        )
        del df

        n_neg = len(wide_df.filter(pl.col("label")==False))
        n_pos = len(wide_df.filter(pl.col("label")==True))

        try:
            scale_pos_weight = int(n_neg / n_pos)
        except ZeroDivisionError:
            scale_pos_weight = 1

        # Update weight balancing
        params["scale_pos_weight"]= scale_pos_weight

        X = wide_df.drop(index_col)
        y_true = wide_df.select("label")
        
        if iteration < 1:
            # fit selector and scaler only once
            X = sel.fit_transform(X, y_true)
            X = sc.fit_transform(X)
            model.fit(X, y_true)
            booster = model.get_booster()

            with open(s.MRMR_FILE, 'wb') as f:
                pickle.dump(sel, f)
            with open(s.SCALER_FILE, 'wb') as f:
                pickle.dump(sc, f)
            with open(s.MODEL_FILE, 'wb') as f:
                pickle.dump(model, f)
            iteration += 1
        else:
            # incremental learning
            try: 
                assert booster is not None
                model.fit(X, y_true, xgb_model=booster)
                mod_name = s.PIPE_DIR / f'iter_{iteration}_xgb.sav'
                if mod_name.exists():
                    print(f"Model {iteration} already stored")
                else:
                    with open(mod_name, 'wb') as f:
                        pickle.dump(model, f)
                iteration += 1
            except AssertionError:
                print(AssertionError)
                print("Model has to be previsouly fitted to continue training from.")

    del X, y_true
    del wide_df
    print(f"XGB model fitted {iteration} time(s).")


    df = pf.pull_features(
            feature_dir=s.FEATURES_DIR,
            label_file=s.LABELS_FILE,
            feature_group="all",
            train_only=False,
            test_only=True,
            step_size=s.PREPROCESSING_KWARGS['segment_eeg']['step_size'],
            inference=True,
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
    del df

    model = pickle.load(open(s.MODEL_FILE, "rb"))
    scaler = pickle.load(open(s.SCALER_FILE, "rb"))
    mrmr = pickle.load(open(s.MRMR_FILE, "rb"))

    X_test = wide_df.drop(index_col)
    X_test_scaled = scaler.transform(X_test)
    X_test_fs = mrmr.transform(X_test_scaled)
    y_pred = model.predict(X_test_fs)


    df_pred = wide_df.select(index_col).with_row_index()
    df_pred = df_pred.with_columns(pl.Series("y_pred", y_pred))
    df_pred.write_parquet(s.PIPE_DIR / f'y_pred_test.parquet')


if __name__ == "__main__":
    main()
    

