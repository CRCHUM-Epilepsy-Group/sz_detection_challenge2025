from szdetect import project_settings as s
from szdetect import pull_features as pf

df = pf.pull_features(
            feature_dir=s.FEATURES_DIR,
            label_file=s.LABELS_FILE,
            feature_group="all",
            train_only=False,
            test_only=True,
            step_size=s.PREPROCESSING_KWARGS['segment_eeg']['step_size']
        )

print(len(df))