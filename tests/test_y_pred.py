import polars as pl 
from pathlib import Path 

home = Path("/mnt/data/SeizureDetectionChallenge2025/data/pretrained_pipe_steps/run_3/")
df = pl.read_parquet(home / "y_pred_test.parquet")

sz_rec = df.filter(pl.col("label")==True).select("unique_id").unique().to_series().to_list()


datasets = df.select("dataset_name").unique().to_series().to_list()

for dataset in datasets:
    df_dt = df.filter(pl.col("dataset_name")==dataset)
    nb_eegs = df_dt.select("unique_id").unique().to_series().to_list()
    subjects = df_dt.select("subject").unique().to_series().to_list()
    sz_eegs = df_dt.filter(pl.col("label")==True).select("unique_id").unique().to_series().to_list()

    print(f"Stats for {dataset}")
    print(f"\tNb EEG files {len(nb_eegs)}")
    print(f"\tNb subjects {len(subjects)}")
    print(f"\tNb EEG files with seizures {len(sz_eegs)}")