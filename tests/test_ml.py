import polars as pl
import numpy as np
import itertools
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import xgboost as xgb
from sklearn.svm import SVC
from pathlib import Path
from szdetect import pull_features as pf
from szdetect import model as mod

models = mod.parse_models("../models.yaml")

models_inst = mod.init_models(models['models'], mod.Model)


df = pf.pull_features(
    feature_dir=Path('./test_data/features'),
    label_file='./test_data/labels.parquet',
    feature_group="efficiency",
    train_only=True)

index_col = ['epoch', 'timestamp', 'dataset_name', 'subject',
             'session', 'run', 'unique_id', 'second', 'label']

feature_col = ['channel', 'freqs', 'feature']

long_df = df.select(index_col + feature_col + ['value'])


long_df = long_df.with_columns([
    pl.col(column).fill_null('missing').alias(column) for column in feature_col
])

long_df = long_df.filter(pl.col('feature') !='band_power')

wide_df = long_df.pivot(
    values='value', 
    index=index_col, 
    on=feature_col,
    maintain_order=True
)

wide_df = wide_df.with_row_index() 
print('long to wide pivot succeeded.')



tau_range = [4, 6]
thresh_range = [0.65, 0.75]
max_depth = [11]
min_child_weight = [7]
# SVM ----------------------------------------
kernel = ['linear', 'rbf']
c = [0.001, 0.01, 0.1, 1]
# XGBoost ------------------------------------
max_depth = [7, 9, 11]
min_child_weight = [5, 7]
# --------------------------------------------
model = SVC()
#model = xgb.XGBClassifier()

def f(x):
    return {'XGBClassifier': (max_depth, min_child_weight),
            'SVC': (kernel, c)
            }[x]



(hyperparam1, hyperparam2) = f(model.__class__.__name__)
combin = [hyperparam1, hyperparam2, [10], [1], tau_range, thresh_range]
all_combinations = list(itertools.product(*combin))

outer_k, inner_k = 3, 2

index_col.append('index')

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



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

from sklearn.base import clone
in_model = clone(model)
hp = hyperparams[0]
if in_model.__class__.__name__ == 'SVC':
        params = {'kernel': hp[0], 'C': hp[1], 'class_weight': 'balanced'}
elif model.__class__.__name__ == 'XGBClassifier':
    params = {'max_depth': hp[0], 'min_child_weight': hp[1],
              'scale_pos_weight': 3000, 'max_delta_step': 1,
              'learning_rate': 0.1, 'gamma': 0.1, 'booster': 'gbtree'}
    
in_model.set_params(**params)
in_model.fit(X_train, y_train)


y_hat = []
ovlp_precision, ovlp_recall, ovlp_f1 = [], [], []
ovlp_FA, ovlp_MA = 0, 0

rec = val_set["unique_id"].unique()[0]

#pred_val = in_model.predict(X_val)                  
val_rec = val_set.filter(pl.col('unique_id')==rec)
record_name = val_rec.select(pl.col('unique_id')).unique().to_series().to_list()[0]

X_val_rec = val_rec.drop(index_columns)
X_val_rec = sc.transform(X_val_rec)

y_pred_val_rec = in_model.predict(X_val_rec)  


#rec_0 = val_rec['index'][0]
#rec_1 = val_rec['index'][-1]

reg_pred = mod.firing_power(y_pred_val_rec, tau=tau_range[0])

step = 1

true_ann = np.array(val_rec['label'])
pred_ann = np.array(reg_pred)
pred_ann = np.array([1 if x >= thresh_range[0] else 0 for x in pred_ann])
OVLP = mod.ovlp(true_ann, pred_ann, step)