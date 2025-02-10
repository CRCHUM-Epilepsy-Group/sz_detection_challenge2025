import itertools
import warnings
import random
import polars as pl
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.base import clone
from feature_engine.selection import MRMR
from sklearn.metrics import precision_recall_fscore_support as prf_scores

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


tau_range = [4]
thresh_range = [0.65]
# XGBoost ------------------------------------
max_depth = [11]
min_child_weight = [7]
reg_alpha = [0.1]
learning_rate = [0.05]
xgb_gamma = [0.5]
model = xgb.XGBClassifier()

def f(x):
    return {
        'XGBClassifier': (max_depth, min_child_weight,
                          reg_alpha, learning_rate, xgb_gamma)
            }[x]

epoch_size = 10
step = 10
# TODO change hyperparams combination to dict and access with keys instead of index
(hyp1, hyp2, hyp3, hyp4, hyp5) = f(model.__class__.__name__)
combin = [hyp1, hyp2, [epoch_size], [step], tau_range, thresh_range, hyp3, hyp4, hyp5]
all_combinations = list(itertools.product(*combin))
random_hyperparams = random.choices(all_combinations, k=10)


data = load_breast_cancer()
X = data.data
y = data.target  # Binary target: 0 = Malignant, 1 = Benign

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_pos = sum(y_train == 1)
n_neg = sum(y_train == 0)
scale_pos_weight = n_neg / n_pos

sc = StandardScaler()
in_model = clone(model)
hp = random_hyperparams[0]
if in_model.__class__.__name__ == 'SVC':
        params = {'kernel': hp[0], 'C': hp[1], 'class_weight': 'balanced'}
elif in_model.__class__.__name__ == 'XGBClassifier':
    params = {'max_depth': hp[0], 'min_child_weight': hp[1],
                  'scale_pos_weight': scale_pos_weight, 'max_delta_step': 1,
                  'eval_metric':'aucpr', 'reg_alpha': hp[6],
                  #'early_stopping_rounds': 200,
                  'learning_rate': hp[7], 'gamma': hp[8], 'booster': 'gbtree'}
    
in_model.set_params(**params)


sel = MRMR(method="FCQ", regression=False)

pipeline = Pipeline([
    ('feature_selector', sel),
    ('scaler', sc),  
    ('classifier', in_model)
])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

print('Pipeline trained')

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

pres_train, rec_train, f1_train, _ = prf_scores(y_train, y_train_pred, 
                                            pos_label=1, average='binary', zero_division=0)
pres_test, rec_test, f1_test, _ = prf_scores(y_test, y_test_pred,
                                            pos_label=1, average='binary',zero_division=0)

print('Scores     \t- Train \t--- Test')
print(f'precision \t- {pres_train:.4f} \t--- {pres_test:.4f}')
print(f'recall    \t- {rec_train:.4f} \t--- {rec_test:.4f}')
print(f'f1-score  \t- {f1_train:.4f} \t--- {f1_test:.4f}')