from pathlib import Path
import datetime
import polars as pl
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import xgboost as xgb
from sklearn.pipeline import Pipeline
from feature_engine.selection import MRMR
from sklearn.preprocessing import StandardScaler
from szdetect import pull_features as pf
# from szdetect import project_settings as s

print('Imports successful')
df = pf.pull_features(
    feature_dir="./test_data/features_v4",
    label_file="./test_data/labels.parquet",
    feature_group="all",
    train_only=True
    )
print('pull_features successful')

index_col = [
    "timestamp",
    "dataset_name",
    "subject",
    "session",
    "run",
    "unique_id",
    "label",
]

feature_col = ["region_side", "freqs", "feature"]

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

print(wide_df.filter(pl.any_horizontal(pl.all().is_null())))

print("preparing data")
X = wide_df.drop(index_col)
y = wide_df.select('label')

model = xgb.XGBClassifier()
sel = MRMR(method="FCQ", regression=False)
sc = StandardScaler()
pipeline = Pipeline([
        ('feature_selector', sel),
        ('scaler', sc),  
        ('classifier', model)
])



print('Training ...')
tt1 = datetime.datetime.now()
# pipeline.fit(X.to_pandas(), y.to_pandas().values.ravel())
X_sel = sel.fit_transform(X.to_pandas(), y.to_pandas().values.ravel())
print("inspecting output of mrmr")
print("nb isna", X_sel.isna().sum().sum())
print("nb isinf", np.isinf(X_sel).sum().sum())

X_sc = sc.fit_transform(X_sel)
print("inspecting output of scaler")
print("nb isna", np.isnan(X_sc).sum().sum())
print("nb isinf", np.isinf(X_sc).sum().sum())

tt2 = datetime.datetime.now()
print(f'\n\t\tTraining time for one model in outer fold is {tt2-tt1}')
