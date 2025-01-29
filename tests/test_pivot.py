import polars as pl
from pathlib import Path
from szdetect import pull_features as pf


df = pf.pull_features(
    feature_dir=Path('./test_data/features'),
    label_file='./test_data/labels.parquet',
    feature_group="all",
    train_only=True)

index_col = ['epoch', 'timestamp', 'dataset_name', 'subject',
             'session', 'run', 'unique_id', 'second', 'label']

feature_col = ['channel', 'freqs', 'feature', 'm', 'r', 'embed_dim',
               'pre_transform', 'wavelet', 'max_level',
               'method', 'region_side', 'params']

long_df = df.select(index_col + feature_col + ['value'])


long_df = long_df.with_columns([
    pl.col(column).fill_null('missing').alias(column) for column in feature_col
])

long_df = long_df.filter(pl.col('feature') !='band_power')

wide_df = long_df.pivot(
    values='value', 
    index=index_col, 
    on=feature_col,
)
