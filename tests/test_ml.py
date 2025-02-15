import polars as pl
import numpy as np
import itertools
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import xgboost as xgb
from sklearn.base import clone
from sklearn.svm import SVC
from pathlib import Path
from sklearn.pipeline import Pipeline
from szdetect import pull_features as pf
from szdetect import model as mod
from feature_engine.selection import MRMR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

# models = mod.parse_models("../models.yaml")

# models_inst = mod.init_models(models['models'], mod.Model)


df = pf.pull_features(
    feature_dir=Path('./test_data/features_v4'),
    label_file="./test_data/labels.parquet",
    feature_group="all",
    #train_only=True,
)

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

long_df = df.select(index_col + feature_col + ["value"])

wide_df = long_df.pivot(
    values="value", index=index_col, on=feature_col, maintain_order=True
)

wide_df = wide_df.with_row_index()
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
max_depth = [3, 5, 7, 9, 11]
min_child_weight = [1, 3, 5, 7]
reg_alpha = [0, 0.01, 0.1, 1, 10, 100]
learning_rate = [0.01, 0.05, 0.1, 0.2]
xgb_gamma = [0.1,  0.3, 0.5]
# --------------------------------------------
# model = SVC()
model = xgb.XGBClassifier()


def f(x):
    return {
        'XGBClassifier': (max_depth, min_child_weight,
                          reg_alpha, learning_rate, xgb_gamma),
            'SVC': (kernel, c, svc_gamma, shrinking, tol)
            }[x]

epoch_size = 10
step = 4
# TODO change hyperparams combination to dict and access with keys instead of index
(hyp1, hyp2, hyp3, hyp4, hyp5) = f(model.__class__.__name__)
combin = [hyp1, hyp2, [epoch_size], [step], tau_range, thresh_range, hyp3, hyp4, hyp5]

all_combinations = list(itertools.product(*combin))

outer_k, inner_k = 3, 2

# index_col.append('index')

outer_k, inner_k = 3, 2
print('Init cross validation')


hyperparams=all_combinations
data=wide_df
index_columns=index_col

clf = model.__class__.__name__

outer_splits = mod.get_train_test_splits(data[['subject']], outer_k)

for i, out_split in enumerate(outer_splits): print(i, out_split)

train_val_set = data.filter(pl.col('subject').is_in(outer_splits[out_split]['train']))
test_set = data.filter(pl.col('subject').is_in(outer_splits[out_split]['test']))

X_test = test_set.drop(index_columns)
y_test = test_set.select('label')


inner_splits = mod.get_train_test_splits(train_val_set[['subject']], inner_k)


for innersplit in inner_splits: print(innersplit)
train_set = train_val_set.filter(pl.col('subject').is_in(inner_splits[innersplit]['train']))
val_set = train_val_set.filter(pl.col('subject').is_in(inner_splits[innersplit]['test']))

X_train = train_set.drop(index_columns)
y_train = train_set.select('label')
X_val = val_set.drop(index_columns)
y_val = val_set.select('label')
#just an example to get data with 2 classes
#for k in range(2,20):
#   y_train[k,0] = True
n_neg = len(train_set.filter(pl.col("label")==False))
n_pos = len(train_set.filter(pl.col("label")==True))
scale_pos_weight = int(n_neg / n_pos)

sel = MRMR(method="FCQ", regression=False) #TODO take feature selector in args


sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_val = sc.transform(X_val)


in_model = clone(model)
hp = hyperparams[0]
if in_model.__class__.__name__ == 'SVC':
    params = {'kernel': hp[0], 'C': hp[1], 'class_weight': 'balanced',
                 'gamma': hp[6], 'shrinking': hp[7], 'tol': hp[8]}
elif in_model.__class__.__name__ == 'LogisticRegression':
    params = {'solver': hp[0], 'C': hp[1], 'class_weight': 'balanced'}
elif in_model.__class__.__name__ == 'DecisionTreeClassifier':
    params = {'splitter': hp[0], 'criterion': hp[1], 'class_weight': 'balanced'}
elif in_model.__class__.__name__ == 'KNeighborsClassifier':
    params = {'algorithm': hp[0], 'n_neighbors': hp[1]}
elif in_model.__class__.__name__ == 'XGBClassifier':
    params = {'max_depth': hp[0], 'min_child_weight': hp[1],
              'scale_pos_weight': scale_pos_weight, 
              'max_delta_step': 1,
              'eval_metric':'aucpr', 'reg_alpha': hp[6],
              'learning_rate': hp[7], 'gamma': hp[8], 'booster': 'gbtree'}
else:
        params = {}
   
    
in_model.set_params(**params)
# in_model.fit(X_train, y_train)
pipeline = Pipeline([
    ('feature_selector', sel),
    ('scaler', sc),  
    ('classifier', in_model)
    ])
pipeline.fit(X_train.to_pandas(), y_train.to_pandas())

print('Training ... ')



y_hat = []
y_val_temp = []
ovlp_precision, ovlp_recall, ovlp_f1 = [], [], []
ovlp_FA, ovlp_MA = 0, 0

rec = val_set["unique_id"].unique()[0]

#pred_val = in_model.predict(X_val)                  
val_rec = val_set.filter(pl.col('unique_id')==rec)
record_name = val_rec.select(pl.col('unique_id')).unique().to_series().to_list()[0]

X_val_rec = val_rec.drop(index_columns)
# X_val_rec = sc.transform(X_val_rec)

# y_pred_val_rec = in_model.predict(X_val_rec)
y_pred_val_rec = pipeline.predict(X_val_rec.to_pandas())


#rec_0 = val_rec['index'][0]
#rec_1 = val_rec['index'][-1]

reg_pred = mod.firing_power(y_pred_val_rec, tau=hp[4])



true_ann = np.array(val_rec['label'])
pred_ann = np.array(reg_pred)
pred_ann = np.array([1 if x >= hp[5] else 0 for x in pred_ann])
OVLP = mod.ovlp(true_ann, pred_ann, step=hp[3])

pred_ann = np.array(pred_ann, dtype=bool)
pres_rec, rec_rec, f1_rec, _ = precision_recall_fscore_support(true_ann, pred_ann, 
                                                               pos_label=1, average='binary',
                                                               zero_division=0)

y_hat.extend(pred_ann)
y_val_temp.extend(true_ann)

y_hat = np.array(y_hat, dtype=bool)
pres_rg, rec_rg, f1_rg, _ = precision_recall_fscore_support(y_val_temp, y_hat,
                                                                     pos_label=1, average='binary',
                                                                     zero_division=0)
# s = len(support_rg)
f1_rg = float("{:.4f}".format(f1_rg))
pres_rg = float("{:.4f}".format(pres_rg))
rec_rg = float("{:.4f}".format(rec_rg))

model = pipeline[-1]
if model.__class__.__name__ in ['LogisticRegression', 'SVC', 'XGBClassifier']:
    y_pred_score = pipeline.predict_proba(X_test.to_pandas())[:, 1]
    y_test = np.array(y_test, dtype=int)
    roc = roc_auc_score(y_test, y_pred_score)  # sample-based
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_score)  # sample-based
else:
    roc = np.nan
    fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])
