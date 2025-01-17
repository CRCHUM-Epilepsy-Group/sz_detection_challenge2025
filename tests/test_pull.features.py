from pathlib import Path
from szdetect import pull_features as pf


df = pf.pull_features(
    feature_dir=Path('./test_data/features'),
    label_file='./test_data/labels.parquet',
    feature_group="efficiency",
    train_only=True)
   