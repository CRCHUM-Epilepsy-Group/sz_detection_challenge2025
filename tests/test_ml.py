import polars as pl
from pathlib import Path
from szdetect import pull_features as pf


# features_file = Path('./test_data/features/tuh_sz_bids_007_00_00.parquet')
# labels_file = Path('./test_data/labels.parquet')
# features = pl.read_parquet(features_file)
# labels = pl.read_parquet(labels_file)

df = pf.pull_features(
    feature_dir=Path('./test_data/features'),
    label_file='./test_data/labels.parquet',
    feature_group="efficiency",
    train_only=True)


