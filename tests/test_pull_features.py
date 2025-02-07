# from pathlib import Path
import polars as pl
from szdetect import pull_features as pf
from szdetect import project_settings as s

print('Imports successful')
df = pf.pull_features(
    feature_dir=s.FEATURES_DIR,
    label_file=s.LABELS_FILE,
    feature_group="all",
    train_only=True)
print('pull_features successful')

index_col = ['dataset_name', 'subject', 'session', 'run',
              'timestamp', 'unique_id', 'label']

feature_col = ['region_side', 'freqs', 'feature']

long_df = df.select(index_col + feature_col + ['value'])

long_df = long_df.with_columns([
    pl.col(column).fill_null('missing').alias(column) for column in feature_col
])

wide_df = long_df.pivot(
    values='value', 
    index=index_col, 
    on=feature_col,
    maintain_order=True
)
print('pivot successful')

print(f'Nbr datasets {len(wide_df.select(pl.col("dataset_name").unique()))}')
print(f'Nbr subjects {len(wide_df.select(pl.col("subject").unique()))}')
print(f'Nbr EEG files {len(wide_df.select(pl.col("unique_id").unique()))}')

n_neg = len(wide_df.filter(pl.col("label")==False))
n_pos = len(wide_df.filter(pl.col("label")==True))
print(f'-> nbr negative epochs {n_neg}')
print(f'-> nbr positive epochs {n_pos}')
print(f'--> scale_pos_weight {int(n_neg/n_pos)}')
