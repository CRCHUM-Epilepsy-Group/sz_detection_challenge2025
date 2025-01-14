import polars as pl
from szdetect import project_settings as s

def nb_seizures(df):
    return df["label"].cast(pl.Int8).diff().abs().sum() // 2

def nb_patients(df):
    return len(df.select(pl.col("subject").unique()).to_series().to_list())


df = pl.read_parquet(s.LABELS_FILE)
summary_df = pl.DataFrame([])

for dataset, _ in s.BIDS_DATASETS.keys():
    sub_df = df.filter((pl.col("dataset_name") == dataset))
    train_df = sub_df.filter((pl.col("training") == True))
    test_df = sub_df.filter((pl.col("training") == False))
    row = {
        "dataset_name": dataset,
        "nbr_subjects": nb_patients(sub_df),
        "total_nb_seizures": nb_seizures(sub_df),
        "sub_train": nb_patients(train_df),
        "sub_test": nb_patients(test_df),
        "sz_train": nb_seizures(train_df),
        "sz_test": nb_seizures(test_df),
    }

    if summary_df is None:
        summary_df = pl.DataFrame([row])
    else:
        summary_df = pl.concat([summary_df, pl.DataFrame([row])], how="vertical")

print(summary_df)
    

