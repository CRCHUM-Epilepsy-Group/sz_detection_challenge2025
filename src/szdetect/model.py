import yaml
from pathlib import Path

#import pandas as pd
import numpy as np
import polars as pl
import scipy.stats as stats
import warnings
import logging
import pickle


with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import xgboost as xgb
import lightgbm as lgb
import yaml
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
#from sklearn import metrics
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler#, OneHotEncoder
#from skopt import BayesSearchCV, dump
from skopt import space
# from sklearn.model_selection import StratifiedGroupKFold
# from sklearn.preprocessing import LabelEncoder
# from skopt.plots import plot_objective
# import matplotlib.pyplot as plt
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

#from eegml import featureengineering as fe
from szdetect import project_settings as s

s.LOGS_FILE.parent.mkdir(exist_ok=True, parents=True)
logging.basicConfig(filename=s.LABELS_FILE, level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

xgb.set_config(verbosity=0)


class Model:
    """Parser class for ML models and hyperaparameter search space"""

    def __init__(self, key: str, args):
        """
        Args:
            key: Key for mapping between YAML config and Python function
            args: Arguments of the corresponding Python function
        """
        self.key = key
        self.args = args
        self.model_map = {
            "real": space.Real,
            "int": space.Integer,
            "loguniform": stats.loguniform,
            "stdscaler": StandardScaler,
            #"hsr": fe.HyperspectralReducer,
            "svm": svm.SVC,
            "enet": SGDClassifier,
            "xgb": xgb.XGBClassifier,
            "lgb": lgb.LGBMClassifier,
            "rf": RandomForestClassifier,
        }

    def get_model(self):
        return self.model_map[self.key](**self.args)


def model_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Model:
    return Model(**loader.construct_mapping(node))


def get_loader():
    """Add constructors to PyYAML loader."""
    loader = yaml.SafeLoader
    loader.add_constructor("!Model", model_constructor)
    return loader


def parse_models(models_file):
    """Read a YAML models file."""

    f = Path(models_file)

    with open(f, "r") as stream:
        try:
            models = yaml.load(stream, Loader=get_loader())
        except yaml.YAMLError as exc:
            print(exc)
            return dict()
    return models


def init_models(in_dict, instance):
    for k, v in in_dict.items():
        if isinstance(v, instance):
            in_dict[k] = v.get_model()
        elif isinstance(v, list):
            in_dict[k] = [
                val.get_model() if isinstance(val, instance) else val for val in v
            ]
    return in_dict


def cross_validate(model, hyperparams, data, k, inner_k,
                   home_path=home):
    """Get CV score with an inner CV loop to select hyperparameters.

    Args:
        model : scikit-learn estimator, for example `LogisticRegression()`
          The base model to fit and evaluate. `model` itself is not modified.
        hyperparams : list[tuple]
          list of possible combinations of the hyperparameters grid.
        data : dataframe
          Subset dataframe of the original data, after outer split.
        k : int
          the number of splits for the k-fold cross-validation.
        inner_k : int
          the number of splits for the nested cross-validation (hyperparameter
          selection).
        home_path : str
          Path to the home directory. Should be modified if working on server.
    Returns:
        scores : list[float]
           The scores obtained for each of the cross-validation folds
    """

    results_rows = []
    df_roc_curves = pl.DataFrame([])

    clf = model.__class__.__name__

    win_size = hyperparams[0][2]
    step = hyperparams[0][3]
    all_scores = []

    for i, (train_idx, test_idx) in enumerate(get_train_test_splits(data[['Patient_ID']], k)):
        
        logger.info(f"\nOuter CV loop: fold {i+1}")

        train_val_set = data.iloc[train_idx]
        test_set = data.iloc[test_idx]

        X_test = test_set.iloc[:, 4:-1]
        # y_test = test_set.iloc[:, -1]

        gridsearch_per_fold = grid_search(model, hyperparams, train_val_set, inner_k,
                                          outer_fold_idx=i,
                                          home_path=home_path)
        best_model = gridsearch_per_fold['best_model']
        best_hp = gridsearch_per_fold['best_hyperparam']
        scaler = gridsearch_per_fold['scaler']

        model_name = f'fold_{i+1}_{clf}_{best_hp[0]}_{best_hp[1]}_win{best_hp[2]}_step{best_hp[3]}.sav'
        pickle.dump(best_model, open(s.RESULTS_DIR / model_name, 'wb'))

        X_test = scaler.transform(X_test)
        
        # Make predictions and calculate performance metrics
        metrics = calculate_metrics(best_model, test_set, scaled_X_test=X_test,
                                    tau=best_hp[4], threshold=best_hp[5], step=best_hp[3],
                                    show=False)
        
        results_rows.append(
            {
            'fold': i+1,
            'train_Patient_ID': list(train_val_set.Patient_ID.unique()),
            'test_Patient_ID': list(test_set.Patient_ID.unique()),
            'model': model_name,
            'max_depth': best_hp[0],
            'min_child_weight': best_hp[1],
            'win_size': best_hp[2],
            'step': best_hp[3],
            'tau': best_hp[4],
            'threshold': best_hp[5],
            'avg_latency': np.nanmean(np.array(metrics['latencies'])),
            'total_false_alarms': metrics['ovlp_FA'],
            'total_missed_alarms': metrics['ovlp_MA'],
            'f1_score_ovlp': metrics['ovlp_f1'],
            'precision_ovlp': metrics['ovlp_precision'],
            'recall_ovlp': metrics['ovlp_recall'],
            'f1_score_regularized': metrics['f1_score_regularized'],
            'roc_auc_score': metrics['roc_auc_score'],
            'latencies': metrics['latencies'],
            'FAR_per_day': metrics['FAR'],
            'Time_in_warning': metrics['Time_in_warning'],
            'percentage_tiw': metrics['percentage_tiw'],
            }
        )

        df_roc = pl.DataFrame(
            {
                'fold': np.ones(len(metrics['roc_fpr'])) * (i+1),
                'roc_fpr': metrics['roc_fpr'],
                'roc_tpr': metrics['roc_tpr'],
                'roc_thresholds': metrics['roc_thresholds']
            }
        )

        logger.info(f"Outer CV loop: finished fold {i+1}, f1-score={metrics['ovlp_f1']}%, ROC-AUC={metrics['roc_auc_score']}")
        all_scores.append(metrics['ovlp_f1'])
        df_roc_curves = pl.concat([df_roc_curves, df_roc], how="vertical")

    df_results = pl.DataFrame(results_rows)

    s.RESULTS_DIR.parent.mkdir(exist_ok=True, parents=True)
    df_results.write_csv(s.RESULTS_DIR / "cv_results.csv")
    df_roc_curves.write_csv(s.RESULTS_DIR / "cv_roc_curves.csv")
    print("Results have been successfully stored.")

    return all_scores