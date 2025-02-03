import polars as pl
import itertools
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import xgboost as xgb
from sklearn.svm import SVC
from pathlib import Path
from szdetect import pull_features as pf
from szdetect import model as mod
from szdetect import project_settings as s

models = mod.parse_models("models.yaml")

models_inst = mod.init_models(models['models'], mod.Model)


df = pf.pull_features(
    feature_dir=s.FEATURES_DIR,
    label_file=s.LABELS_FILE,
    feature_group="all",
    train_only=True)

index_col = ['epoch', 'timestamp', 'dataset_name', 'subject',
             'session', 'run', 'unique_id', 'second', 'label']

feature_col = ['channel', 'freqs', 'feature']

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

wide_df = wide_df.with_row_index() 
print('long to wide pivot succeeded.')



tau_range = [4]
thresh_range = [0.65]
# SVM ----------------------------------------
kernel = ['linear', 'rbf']
c = [0.001, 0.01, 0.1, 1]
# XGBoost ------------------------------------
max_depth = [7, 11]
min_child_weight = [5, 7]
# --------------------------------------------
#model = SVC()
model = xgb.XGBClassifier()

def f(x):
    return {'XGBClassifier': (max_depth, min_child_weight),
            'SVC': (kernel, c)
            }[x]


epoch_size = s.PREPROCESSING_KWARGS['segment_eeg']['window_duration']
step = s.PREPROCESSING_KWARGS['segment_eeg']['step_size']
(hyperparam1, hyperparam2) = f(model.__class__.__name__)
combin = [hyperparam1, hyperparam2, [epoch_size], [step], tau_range, thresh_range]
all_combinations = list(itertools.product(*combin))

index_col.append('index')

outer_k, inner_k = 2, 2
print('Init cross validation')
scores = mod.cross_validate(model = model,
                            hyperparams=all_combinations,
                            data=wide_df,
                            k=outer_k, inner_k=inner_k,
                            index_columns=index_col, 
                            #feature_group='efficiency'
)