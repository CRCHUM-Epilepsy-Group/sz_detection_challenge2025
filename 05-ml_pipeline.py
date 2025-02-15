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

# models = mod.parse_models("models.yaml")

# models_inst = mod.init_models(models['models'], mod.Model)


df = pf.pull_features(
    feature_dir=s.FEATURES_DIR,
    label_file=s.LABELS_FILE,
    feature_group="all",
    train_only=True,
    num_eegs=2500
)

index_col = [
    "dataset_name",
    "subject",
    "session",
    "run",
    "unique_id",
    "timestamp",
    "second",
    "label",
]

feature_col = ["region_side", "freqs", "feature"]

long_df = df.select(index_col + feature_col + ["value"])

wide_df = long_df.pivot(
    values="value", index=index_col, on=feature_col, maintain_order=True
)

# wide_df = wide_df.with_row_index()
print("long to wide pivot succeeded.")


tau_range = [4]
thresh_range = [0.65]
# SVM ----------------------------------------
kernel = ["linear", "rbf"]
c = [0.001, 0.01, 0.1, 1]
svc_gamma = ['scale', 0.001, 0.01, 0.1, 1]
shrinking=  [True, False]
tol =[1e-5, 1e-4, 1e-3, 1e-2]
# XGBoost ------------------------------------
max_depth = [ 5, 7, 9, 11]
min_child_weight = [ 9, 11, 13, 15, 17]
reg_alpha = [1, 10, 50, 100, 200]
learning_rate = [0.01, 0.05, 0.1, 0.2]
xgb_gamma = [0.3, 0.5, 1, 10]
# --------------------------------------------
# model = SVC()
model = xgb.XGBClassifier()


def f(x):
    return {
        'XGBClassifier': (max_depth, min_child_weight,
                          reg_alpha, learning_rate, xgb_gamma),
            'SVC': (kernel, c, svc_gamma, shrinking, tol)
            }[x]


epoch_size = s.PREPROCESSING_KWARGS['segment_eeg']['window_duration']
step = s.PREPROCESSING_KWARGS['segment_eeg']['step_size']
# TODO change hyperparams combination to dict and access with keys instead of index
(hyp1, hyp2, hyp3, hyp4, hyp5) = f(model.__class__.__name__)
combin = [hyp1, hyp2, [epoch_size], [step], tau_range, thresh_range, hyp3, hyp4, hyp5]

all_combinations = list(itertools.product(*combin))

# index_col.append("index")

outer_k, inner_k = 3, 5
print('Init cross validation')
scores = mod.cross_validate(model = model, #TODO replace "xgb"
                            hyperparams=all_combinations,
                            nb_rand_hp=50,
                            data=wide_df,
                            k=outer_k, inner_k=inner_k,
                            index_columns=index_col, 
                            #feature_group='efficiency'
)
