import os
import pandas as pd
from szdetect import project_settings as s

features_dir = s.FEATURES_DIR
labels_file = s.LABELS_FILE
df_labels = pd.read_parquet(labels_file)

sz_files = df_labels[df_labels.label == True].unique_id.unique()
sz_files_train  = df_labels[
    (df_labels.label == True) & (df_labels.training == True)
    ].unique_id.unique()
sz_files_test = df_labels[
    (df_labels.label == True) & (df_labels.training == False)
    ].unique_id.unique()
exists_train = [os.path.exists(os.path.join(features_dir, file+'.parquet')) for file in sz_files_train]
exists_test = [os.path.exists(os.path.join(features_dir, file+'.parquet')) for file in sz_files_test]


# Get only the files that exist
existing_files_train = [file for file, exist in zip(sz_files_train, exists_train) if exist]
existing_files_test = [file for file, exist in zip(sz_files_test, exists_test) if exist]


print("existing_files_train \n", existing_files_train)
print("existing_files_test \n", existing_files_test)



