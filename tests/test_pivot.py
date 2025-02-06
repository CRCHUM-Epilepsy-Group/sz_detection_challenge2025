import polars as pl
from pathlib import Path
from szdetect import pull_features as pf
from szdetect import project_settings as s


df = pf.pull_features(
    feature_dir=s.FEATURES_DIR,
    label_file=s.LABELS_FILE,
    feature_group="all",
    train_only=True)

print('Features pulled')

index_col = ['epoch', 'timestamp', 'dataset_name', 'subject',
             'session', 'run', 'unique_id', 'second', 'label']

feature_col = ['channel', 'freqs', 'feature']

long_df = df.select(index_col + feature_col + ['value'])


long_df = long_df.with_columns([
    pl.col(column).fill_null('missing').alias(column) for column in feature_col
])

#long_df = long_df.filter(pl.col('feature') !='band_power')
#print('band_power features filtered out')

wide_df = long_df.pivot(
    values='value', 
    index=index_col, 
    on=feature_col,
)
print('Long to wide pivot succeeded.')

print('Nb of FALSE labels', len(wide_df.filter(pl.col('label')==False)))
print('Nb of TRUE labels', len(wide_df.filter(pl.col('label')==True)))