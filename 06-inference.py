import pickle
import polars as pl
from pathlib import Path
from szdetect import pull_features as pf
from szdetect import project_settings as s


df = pf.pull_features(
    feature_dir=s.FEATURES_DIR,
    label_file=s.LABELS_FILE,
    feature_group="all",
    train_only=True)

index_col = ['dataset_name', 'subject', 'session', 'run',
              'timestamp', 'label']

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

# wide_df = wide_df.with_row_index() 
print('long to wide pivot succeeded.')

# TODO add filepath to config and project_settings and load (s.MODEL_FILE)
model_file = s.MODEL_FILE
pretrained_model = pickle.load(open(model_file, 'rb'))


X_test = wide_df.drop(index_col)
# X_test = scaler.transform(X_test_rec) # TODO add scaling on training data
y_pred = pretrained_model.predict(X_test)