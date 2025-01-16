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
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
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

def calculate_metrics(model, test_set, scaled_X_test,
                      tau=6, threshold=.85, step=1,
                      show=True, home_path=home):
    """
    Estimate class of the input test set samples given a trained model,
    implement regularization using firing power method and evaluate performance.
    Args:
        model        : trained model for predictions
        test_set     : test or validation set to be used
        scaled_X_test: scaled testing features
        tau          : (regularization hyperparameter) window size for averaging firing power
        threshold    : (regularization hyperparameter) minimum fraction of "full" firing power to raise an alarm
        step         : step that was used for feature extraction
        show         : bool - if True, plots features and prediction per seizure in the given set
        home_path    : str, home directory for the EDF and features
    Returns:
        dict of metrics to evaluate the model
    """

    y_test = test_set.iloc[:, -1]
    latencies = []
    TP_SZ_overlap = []
    y_hat = []
    ovlp_precision, ovlp_recall, ovlp_f1 = [], [], []
    ovlp_FA, ovlp_MA = 0, 0
    FP_h, TP_h, FN_h = [], [], []
    TP_duration, FN_duration = [], []
    tiw, percentage_tiw = [], []
    FAR = []

    for i in test_set.Record_index.unique():
        # Predictions per record _______________________________________________________________________________________
        pred = predictions_per_record(test_set, record_row=i, model=model,
                                      scaled_X_test=scaled_X_test,
                                      tau=tau, threshold=threshold, step=step,
                                      plot_alarms=show, show_plots=show,
                                      home_path=home_path)  # ;

        # Event-based metrics
        OVLP = pred['OVLP']
        ovlp_precision.append(OVLP['precision'])
        ovlp_recall.append(OVLP['recall'])
        ovlp_f1.append(OVLP['f1'])
        ovlp_FA += OVLP['FP']
        ovlp_MA += OVLP['FN']

        latencies.extend(pred['Detection_latency'])
        TP_SZ_overlap.extend(pred['TP_SZ_overlap'])
        FAR.append(pred['FAR'])
        #FP_h.extend(pred['FP_hours'])
        #TP_h.extend(pred['TP_hours'])
        #FN_h.extend(pred['FN_hours'])
        tiw.extend(pred['Time_in_warning'])
        #TP_duration.extend(pred['TP_duration'])
        #FN_duration.extend(pred['FN_duration'])
        percentage_tiw.append(pred['percentage_tiw'])
        y_hat.extend(pred['regularized_predictions'])

    # L = [1 if x >= threshold else 0 for x in y_hat]
    # sample-based, on all test dataset
    try:
        assert len(y_test) == len(y_hat)
        pres_rg, rec_rg, f1_rg, support_rg = precision_recall_fscore_support(y_test, y_hat,
                                                                             zero_division=0)
        s = len(support_rg)
        f1_rg = 100 * float("{:.4f}".format(f1_rg[s - 1]))
        pres_rg = 100 * float("{:.4f}".format(pres_rg[s-1]))
        rec_rg = 100 * float("{:.4f}".format(rec_rg[s-1]))

        if model.__class__.__name__ in ['LogisticRegression', 'SVC', 'XGBClassifier']:
            # y_pred_score = model.decision_function(scaled_X_test)
            # roc = roc_auc_score(y_test, y_pred_score)  # sample-based
            y_pred_score = model.predict_proba(scaled_X_test)[:, 1]
            roc = roc_auc_score(y_test, y_pred_score)  # sample-based
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_score)  # sample-based
        else:
            roc = np.nan
            fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])

    except (AssertionError, ValueError) as error:
        logger.error(f'ERROR {error} len(true_ann)={len(y_test)}, len(pred_ann)={len(y_hat)}',
                     exc_info=True)
        f1_rg = np.nan
        pres_rg = np.nan
        rec_rg = np.nan
        roc = np.nan
        fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])

    return {'f1_score_regularized': f1_rg,  # sample-based
            'precision_regularized': pres_rg,  # sample-based
            'recall_regularized': rec_rg,  # sample-based
            'roc_auc_score': float("{:.4f}".format(roc)),  # sample-based
            'roc_fpr': fpr, 'roc_tpr': tpr, 'roc_thresholds': thresholds,  # sample-based
            'ovlp_precision': 100 * float("{:.4f}".format(np.nanmean(ovlp_precision))),  # event-based
            'ovlp_recall': 100 * float("{:.4f}".format(np.nanmean(ovlp_recall))),  # event-based
            'ovlp_f1': 100 * float("{:.4f}".format(np.nanmean(ovlp_f1))),  # event-based
            'ovlp_FA': ovlp_FA, 'ovlp_MA': ovlp_MA,  # event-based
            'latencies': latencies,  # event-based
            #'TP_SZ_overlap': TP_SZ_overlap,  # event-based
            'FAR': FAR,  # event-based
            'Time_in_warning': tiw, 'percentage_tiw': percentage_tiw,  # event-based
            #'TP_duration': TP_duration, 'FN_duration': FN_duration,  # event-based
            #'FP_hours': FP_h, 'TP_hours': TP_h, 'FN_hours': FN_h  # event-based
            }

def cross_validate(model, hyperparams:list, data:pl.DataFrame, k:int, inner_k:int,
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
    feature_columns = []  # TODO get feature column names
    # nonfeature_columns = ['unique_id', 'subject', 'session', 'run', 'dataset_name', 'second', 'timestamp' 'label', 'training']
        

    #win_size = hyperparams[0][2]
    #step = hyperparams[0][3]
    all_scores = []

    for i, (train_idx, test_idx) in enumerate(get_train_test_splits(data[['Patient_ID']], k)):
        
        logger.info(f"\nOuter CV loop: fold {i+1}")

        train_val_set = data[train_idx]
        test_set = data[test_idx]

        X_test = test_set.select([pl.col(col) for col in feature_columns])
        # or exclude non-features columns
        #X_test = test_set.select([col for col in test_set.columns if col not in nonfeature_columns])

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
                                    tau=best_hp[4], threshold=best_hp[5],
                                    show=False)
        
        results_rows.append(
            {
            'fold': i+1,
            'train_Patient_ID': train_val_set.select(pl.col("subject").unique()).to_series().to_list(),
            'test_Patient_ID': test_set.select(pl.col("subject").unique()).to_series().to_list(),
            'model': model_name,
            'max_depth': best_hp[0],
            'min_child_weight': best_hp[1],
            #'win_size': best_hp[2],
            #'step': best_hp[3],
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