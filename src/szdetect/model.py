import yaml
from pathlib import Path

#import pandas as pd
import numpy as np
import polars as pl
import scipy.stats as stats
import warnings
import logging
import pickle
from random import shuffle
from sklearn.base import clone


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


def split_subjects(subjects_list:list, k):
    """
    Split a given list of subjects into k groups of subjects for train/test split folds.
    This split does not account for number of seizures per subject.
    Args:
        subjects_list: list of subject IDs.
        k: int
         number of folds.
    Returns:
        dictionary of folds, containing train and test subjects shuffled groups.
    """
    
    shuffle(subjects_list)
    folds = {}
    sl = [subjects_list[i::k] for i in range(k)]
    for i in range(k):
        folds[f'fold{i}'] = {'test': sl[i], 'train': sum(sl[0:i]+sl[i+1:], [])}
    return folds


def get_train_test_splits(subjects_df, k: int):
    """
    Split dataframe into train and test split folds according to subjects, given K folds.
    Args:
        subjects_df: dataframe of containing a column 'subject'
        k: number of folds
    Returns:
        List of k tuples containing train and test indexes of the dataframe, for k folds.
    """
    idx = []
    subjects_list = subjects_df.select(pl.col("subject").unique()).to_series().to_list()

    splits = split_subjects(subjects_list, k=k)

    for i in range(k):
        train_idx = np.array(
            subjects_df.filter(
                pl.col('subject').is_in(splits[f'fold{i}']['train'])
            ).select(pl.col("index")).to_series()
        )
        test_idx = np.array(
            subjects_df.filter(
                pl.col('subject').is_in(splits[f'fold{i}']['test'])
            ).select(pl.col("index")).to_series()
        )
        idx.append((train_idx, test_idx))
    del train_idx, test_idx
    return idx


def predictions_per_record(test_dataset, 
                           record_id: str,
                           model, scaled_X_test,
                           tau=6, threshold=0.85, step=1,
                           #plot_alarms=False, show_plots=True, save_fig=None, 
                           #home_path=home
                           ):
    """
    Generate predictions on the specified record given a trained model, and calculate performance metrics.
    Args:
        test_dataset : data with test seizures
        record_row   : record index in data
        model        : trained model to evaluate
        scaled_X_test: test features scaled using mean of train data
        tau          : window size for firing power regularization
        threshold    : threshold for firing power regularization
        step         : step that was used for moving window feature extraction
        plot_alarms  : bool, plot vertical lines when firing power average crosses the threshold
        show_plots   : bool, plot the features and predictions for 1h preictal of the selected seizure
        save_fig     : str, default=None, is path provided, save the plots as png.
        home_path    : str, home directory for the EDF and features
    Returns:
        dictionary with performances measures of the selected record.
        {
        OVLP: dictionary of TP, FP, FN, precision, recall and f1-score (event-based)
        Detection_latency: latency between seizure onset and start of true positive event
        TP_SZ_overlap: overlap between seizure and true positive event
        Time_in_warning: duration of false positive events
        percentage_tiw: (total time in warning)/(record duration)
        TP_duration: duration of true positive events
        FN_duration: duration of false negative events (missed seizures)
        FP_hours: hours of the day of the false alarms
        TP_hours: hours of the day of the true alarms
        FN_hours: hours of the day of the missed alarms/seizures
        regularized_predictions: array of regularized predictions representing high-risk alarms
        }
    """

    predictions = model.predict(scaled_X_test)

    #fs = 256
    test_rec = test_dataset.filter(pl.col('unique_id')==record_id)
    record_name = test_rec.select(pl.col('unique_id')).unique().to_series().to_list()[0]
    
    logger.info(f'Analyzing EEG record : {record_name}')
    """
    meas_date = datetime.datetime.strptime(records.Record_start_date[record_row],
                                           '%Y-%m-%d %H:%M:%S%z')
    record_duration = records.n_records[record_row]
    rec_dt = datetime.datetime.combine(meas_date.date(), datetime.datetime.min.time())
    rec_dt = rec_dt.replace(tzinfo=pytz.utc).astimezone(pytz.utc)
    diff = meas_date - rec_dt
    tm = test_rec['timestamp']
    tm = np.array(tm) / fs + int(diff.total_seconds())
    """
    rec_idx = test_rec.index.tolist()
    test_lst = test_dataset.index.tolist()
    rec_0 = test_lst.index(rec_idx[0])
    rec_1 = test_lst.index(rec_idx[-1])

    # Regularize prediction output with firing power method
    reg_pred = firing_power(predictions[rec_0:rec_1 + 1], tau=tau)

    """
    Desc = records.Descriptions[record_row]
    Desc = Desc.replace("[", "").replace("]", "").replace("'", "").split(', ') if not (pd.isna(Desc)) else []
    Onsets = records.Onsets[record_row]
    Onsets = Onsets.replace("[", "").replace("]", "").split(', ') if not (pd.isna(Onsets)) else []
    Onsets = [int(x) for x in Onsets]
    Durations = records.Durations[record_row]
    Durations = Durations.replace("[", "").replace("]", "").split(', ') if not (pd.isna(Durations)) else []
    Durations = [int(x) for x in Durations]
    """
    true_ann = np.array(test_rec.annotation)
    pred_ann = np.array(reg_pred)
    pred_ann = np.array([1 if x >= threshold else 0 for x in pred_ann])
    OVLP = ovlp(true_ann, pred_ann, step)
    """
    # ########################################################################################################
    # DO NOT REMOVE COMMENTED LINES IN THIS SECTION
    # EXTRA PLOTS ALLOW VERIFYING DETAILS
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M:%S', time.gmtime(ms)))
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set(xlabel="Time (UTC)", title="True vs predicted labels for patient %a, record %s" % (p, record_row))
    # ax.plot(tm, test_rec['annotation'], c='g', label='y_true')
    ax.scatter(tm, predictions[rec_0:rec_1 + 1], label='y_pred', alpha=.3, linewidths=.3)
    # ax.plot(tm, reg_pred, marker='.', label='y_pred regularized (Firing Power)', alpha=.3)
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Alarm Threshold')

    pe = OVLP['pred_events']
    for e in pe:
        ev0 = test_rec.iloc[pe[e][0]]['timestamp'] / fs + int(diff.total_seconds())
        ev1 = test_rec.iloc[pe[e][-1]]['timestamp'] / fs + int(diff.total_seconds())
        if e in OVLP['pred_sz_events']:
            ax.axvspan(ev0, ev1, alpha=.5, color='green', label='True alarm')
        elif e in OVLP['pred_fp_events']:
            ax.axvspan(ev0, ev1, alpha=.5, color='orange', label='False alarm')

    for s in range(len(Desc)):
        sz_onset = Onsets[s] + int(diff.total_seconds())
        sz_end = Onsets[s] + Durations[s] + int(diff.total_seconds())
        if Desc[s] == 'FBTCS':
            ax.axvspan(sz_onset, sz_end, alpha=0.5, color='red', label='FBTCS')
        elif Desc[s] == 'FIAS':
            ax.axvspan(sz_onset, sz_end, alpha=0.5, color='skyblue', label='FIAS')
        elif Desc[s] == 'FAS':
            ax.axvspan(sz_onset, sz_end, alpha=0.5, color='grey', label='FAS')
        elif Desc[s] == 'FUAS':
            ax.axvspan(sz_onset, sz_end, alpha=0.5, color='powderblue', label='FUAS')

    ax.xaxis.set_major_formatter(formatter)

    if plot_alarms:
        a_pred = np.array(reg_pred) - threshold
        asc_zc = np.where(np.diff(np.sign(a_pred)) > 0)[0]  # crossing threshold in an ascending way
        for xv in asc_zc:
            plt.axvline(x=tm[xv + 1], color='g', linestyle='dotted', label='alarm')
    # legend_without_duplicate_labels(ax)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    if save_fig:
        plt.savefig(save_fig + '_pred.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    """
    """
    latency, overlap = [], []
    tiw = []
    for i in OVLP['pred_sz_events']:
        l = []  # Detection latency between TP event and all available seizure on the record
        w = []  # Overlap between true seizure and detection event
        ev0 = test_rec.iloc[OVLP['pred_events'][i][0]]['timestamp'] / fs
        ev1 = test_rec.iloc[OVLP['pred_events'][i][-1]]['timestamp'] / fs
        for j in range(len(Onsets)):
            l.append(ev0 - Onsets[j])
            if (ev0 > Onsets[j]) and (ev0 < Onsets[j] + Durations[j]):
                # To avoid accounting overlap if seizure is detected slightly after seizure offset
                w.append(min(Onsets[j] + Durations[j], ev1) - ev0)
        l = np.abs(l)
        w = np.abs(w)
        # Detection latency
        latency.append(min(l))  # min is to find the correct seizure associated to the true positive event
        # Overlap between true seizure reference and correct predicted event
        overlap.extend(w)
 
    for i in OVLP['pred_fp_events']:
        ev0 = test_rec.iloc[OVLP['pred_events'][i][0]]['timestamp'] / fs
        ev1 = test_rec.iloc[OVLP['pred_events'][i][-1]]['timestamp'] / fs
        tiw.append(ev1 - ev0)  # TiW Time in warning is duration of FP

    percentage_tiw = np.sum(tiw) / record_duration * 100
    percentage_tiw = round(percentage_tiw, 4)
    # False alarm rate per day = nbr false alarms in record *24h / record duration
    FAR = len(tiw) * 24 * 3600 / record_duration
    FAR = round(FAR, 4)
    """
    #  Sample-based metrics to estimate performance on records that contain no seizures.
    try:
        assert len(true_ann) == len(pred_ann)
        pres_rec, rec_rec, f1_rec, support_rec = precision_recall_fscore_support(true_ann, pred_ann,
                                                                                 zero_division=0)
        supp = len(support_rec)
        msg = f"""Sample-based performance on record {record_row}:  \
        f1={100 * float('{:.4f}'.format(f1_rec[supp-1]))}% \
        precision={100 * float('{:.4f}'.format(pres_rec[supp-1]))}% \
        recall={100 * float('{:.4f}'.format(rec_rec[supp-1]))}%"""
        ## Tiw = {round(np.sum(tiw)/3600, 2)}h/{round(record_duration/3600,1)}h = {percentage_tiw}%
        logger.info(msg)
    except (AssertionError, ValueError) as error:
        logger.error(f'ERROR {error} len(true_ann)={len(true_ann)}, len(pred_ann)={len(pred_ann)}',
                     exc_info=True)

    return {'OVLP': OVLP,
            #'Detection_latency': latency,
            #'TP_SZ_overlap': overlap,
            #'FAR': FAR,
            #'Time_in_warning': tiw,
            #'percentage_tiw': percentage_tiw,
            #'FP_hours': FP_hours, 'TP_hours': TP_hours, 'FN_hours': FN_hours,
            #'TP_duration': TP_duration, 'FN_duration': FN_duration,
            'regularized_predictions': pred_ann}


def calculate_metrics(model, test_set, scaled_X_test,
                      tau=6, threshold=.85, step=1,
                      show=True, 
                      #home_path=home
                      ):
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

    #y_test = test_set.iloc[:, -1]
    y_test = test_set.select('label')
    #latencies = []
    #TP_SZ_overlap = []
    y_hat = []
    ovlp_precision, ovlp_recall, ovlp_f1 = [], [], []
    ovlp_FA, ovlp_MA = 0, 0
    #FP_h, TP_h, FN_h = [], [], []
    #TP_duration, FN_duration = [], []
    #tiw, percentage_tiw = [], []
    #FAR = []

    for rec in test_set["unique_id"].unique():  #This should be index
        
        # Predictions per record _______________________________________________________________________________________
        pred = predictions_per_record(test_set, record_id=rec, model=model,
                                      scaled_X_test=scaled_X_test,
                                      tau=tau, threshold=threshold, step=step,
                                      plot_alarms=show, show_plots=show,
                                      #home_path=home_path
                                      )

        # Event-based metrics
        OVLP = pred['OVLP']
        ovlp_precision.append(OVLP['precision'])
        ovlp_recall.append(OVLP['recall'])
        ovlp_f1.append(OVLP['f1'])
        ovlp_FA += OVLP['FP']
        ovlp_MA += OVLP['FN']

        #latencies.extend(pred['Detection_latency'])
        #TP_SZ_overlap.extend(pred['TP_SZ_overlap'])
        #FAR.append(pred['FAR'])
        #FP_h.extend(pred['FP_hours'])
        #TP_h.extend(pred['TP_hours'])
        #FN_h.extend(pred['FN_hours'])
        #tiw.extend(pred['Time_in_warning'])
        #TP_duration.extend(pred['TP_duration'])
        #FN_duration.extend(pred['FN_duration'])
        #percentage_tiw.append(pred['percentage_tiw'])
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
            #'latencies': latencies,  # event-based
            #'TP_SZ_overlap': TP_SZ_overlap,  # event-based
            #'FAR': FAR,  # event-based
            #'Time_in_warning': tiw, 'percentage_tiw': percentage_tiw,  # event-based
            #'TP_duration': TP_duration, 'FN_duration': FN_duration,  # event-based
            #'FP_hours': FP_h, 'TP_hours': TP_h, 'FN_hours': FN_h  # event-based
            }


def fit_and_score(model, hp, data, train_idx, test_idx,
                  #home_path=home
                  feature_list
                  ):
    """Fit a model on training data and compute its score on test data.
    Args:
        model : scikit-learn estimator (will not be modified)
          the estimator to be evaluated
        hp : tuple
         Combination of hyperparameters to use when fitting the model.
        data : dataframe
          Subset dataframe of the original data, after outer split.
        train_idx : sequence of ints
          the indices of training samples (row indices of X)
        test_idx : sequence of ints
          the indices of testing samples
        home_path : str
          Home directory for the EDF and features
    Returns:
      dictionary of the performance metrics on test data
    """

    train_set = data[train_idx]
    test_set = data[test_idx]
    # In inner fold, test means validation set in the corresponding CV fold.

    #X_train = train_set.iloc[:, 4:-1]
    X_train = train_set.select([pl.col(col) for col in feature_list])
    #y_train = train_set.iloc[:, -1]
    y_train = train_set.select('label')
    #X_test = test_set.iloc[:, 4:-1]
    X_test = test_set.select([pl.col(col) for col in feature_list])

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = clone(model)
    if model.__class__.__name__ == 'SVC':
        params = {'kernel': hp[0], 'C': hp[1], 'class_weight': 'balanced'}
    elif model.__class__.__name__ == 'LogisticRegression':
        params = {'solver': hp[0], 'C': hp[1], 'class_weight': 'balanced'}
    elif model.__class__.__name__ == 'DecisionTreeClassifier':
        params = {'splitter': hp[0], 'criterion': hp[1], 'class_weight': 'balanced'}
    elif model.__class__.__name__ == 'KNeighborsClassifier':
        params = {'algorithm': hp[0], 'n_neighbors': hp[1]}
    elif model.__class__.__name__ == 'XGBClassifier':
        params = {'max_depth': hp[0], 'min_child_weight': hp[1],
                  'scale_pos_weight': 3000, 'max_delta_step': 1,
                  'learning_rate': 0.1, 'gamma': 0.1, 'booster': 'gbtree'}
        # NOTE: The ratio neg/pos can be recalculate in each fold and used for scale_pos_weight 
        # This could be optimized, but I doubt this will have a significant impact on results
    #   testing with 1 hour preictal, ratio was 33 sneg : 1spos
    #   testing on all data, ratio is 5696978 sneg : 1949 spos =(approx.) 2923
    else:
        params = {}

    model.set_params(**params)
    model.fit(X_train, y_train)

    metrics = calculate_metrics(model, test_set, scaled_X_test=X_test,
                                tau=hp[4], threshold=hp[5], step=hp[3],
                                show=False,
                                #home_path=home_path
                                )
    
    score = metrics['f1_score_regularized']

    logger.info(f"\n\tInner CV loop: fit and evaluate one model; f1-score={score}%, ROC={metrics['roc_auc_score']}")
    if model.__class__.__name__ == 'SVC':
        logger.info(f'kernel:{hp[0]}, C:{hp[1]}, win_size:{hp[2]}, step:{hp[3]}, tau:{hp[4]}, threshold:{hp[5]}')
    elif model.__class__.__name__ == 'LogisticRegression':
        logger.info(f'solver:{hp[0]}, C:{hp[1]}, win_size:{hp[2]}, step:{hp[3]}, tau:{hp[4]}, threshold:{hp[5]}')
    elif model.__class__.__name__ == 'DecisionTreeClassifier':
        logger.info(f'splitter:{hp[0]}, criterion:{hp[1]}, win_size:{hp[2]}, step:{hp[3]}, tau:{hp[4]},\
        threshold:{hp[5]}')
    elif model.__class__.__name__ == 'KNeighborsClassifier':
        logger.info(f'algorithm:{hp[0]}, n_neighbors:{hp[1]}, win_size:{hp[2]}, step:{hp[3]}, tau:{hp[4]},\
        threshold:{hp[5]}')
    elif model.__class__.__name__ == 'XGBClassifier':
        logger.info(f'max_depth :{hp[0]}, min_child_weight :{hp[1]}, win_size:{hp[2]},\
        step:{hp[3]}, tau:{hp[4]}, threshold:{hp[5]}')

    return metrics


def grid_search(model, hyperparams, data,
                inner_k: int, outer_fold_idx,
                feature_list
                #home_path=home
                ):
    """
    Exhaustive grid search over hyperparameter combinations for the specified estimator.
    Used for the inner loop of a nested cross-validation.
    The best hyperparameters selected after the cross validation are used to refit the estimator.
    Args:
        model : scikit-learn estimator
          The base estimator, copies of which are trained and evaluated. `model` itself is not modified.
        hyperparams : list[tuple]
          list of possible combinations of the hyperparameters grid.
        data : dataframe
          Subset dataframe of the original data (train/val set), after outer split.
        inner_k : int
          number of inner cross-validation folds
        outer_fold_idx : int
          Index of fold in outer cross-validation loop. Only for reference in log output.
        home_path : str
          Path to the home directory. Should be modified if working on server.
    Returns:
        best_model : scikit-learn estimator
          A copy of `model`, fitted on the whole `(X, y)` data, with the best (estimated) hyperparameter.
    """
    # X: numpy array of shape(n_samples, n_features) the design matrix
    # y: numpy array of shape(n_samples, n_outputs) or (n_samples,) the target vector
    # X and y are from train_val set after split for outer folds.
    inner_cv_results_rows = []

    #win_size = hyperparams[0][2]
    #step = hyperparams[0][3]
    clf = model.__class__.__name__

    all_scores = []
    FA_hp = []
    MA_hp = []
    precision_hp = []
    recall_hp = []
    f1_ovlp_hp = []
    roc_auc_hp = []
    latencies_hp = []
    #TP_SZ_overlap_hp = []
    far_hp = []
    tiw_hp = []
    percentage_tiw_hp = []
    for j, hp in enumerate(hyperparams):
        logger.info(f"\n  Grid search: evaluate hyperparameters = \
        solver:{hp[0]}, C:{hp[1]}, win_size:{hp[2]}, step:{hp[3]}, tau:{hp[4]}, threshold:{hp[5]} ")

        scores_hp = []
        for train_idx, val_idx in get_train_test_splits(data[['subject']], inner_k):
            metrics = fit_and_score(model, hp, data,
                                    train_idx, val_idx,
                                    feature_list
                                    #home_path=home_path
                                    )

            score = metrics['f1_score_regularized']  # sample-based
            scores_hp.append(score)

            # Event-based metrics
            FA_hp.append(metrics['ovlp_FA'])
            MA_hp.append(metrics['ovlp_MA'])
            precision_hp.append(metrics['ovlp_precision'])
            recall_hp.append(metrics['ovlp_recall'])
            f1_ovlp_hp.append(metrics['ovlp_f1'])
            roc_auc_hp.append(metrics['roc_auc_score'])
            latencies_hp.append(np.nanmean(np.array(metrics['latencies'])))
            #TP_SZ_overlap_hp.append(np.nanmean(np.array(metrics['TP_SZ_overlap'])))
            far_hp.append(np.nanmean(np.array(metrics['FAR'])))
            tiw_hp.append(np.nanmean(np.array(metrics['Time_in_warning'])))
            percentage_tiw_hp.append(np.nanmean(np.array(metrics['percentage_tiw'])))

        # NOTE: scoring used for hyperparameter tuning is f1-score calculated as sample-based.
        # DO NOT use regularized f1-score to optimize hyperparameter search
        all_scores.append(np.nanmean(scores_hp))

        inner_cv_results_rows.append(
            {
                'fold': outer_fold_idx + 1,
                'train_subject': data.select(pl.col("subject").unique()).to_series().to_list(),
                'max_depth': hp[0],
                'min_child_weight': hp[1],
                #'win_size': hp[2],
                #'step': hp[3],
                'tau': hp[4],
                'threshold': hp[5],
                'avg_latency': np.nanmean(latencies_hp),
                'total_false_alarms': np.nanmean(FA_hp),
                'total_missed_alarms': np.nanmean(MA_hp),
                'f1_score_regularized': np.nanmean(scores_hp),
                'f1_score_ovlp': np.nanmean(f1_ovlp_hp),
                'roc_auc': np.nanmean(roc_auc_hp),
                'precision_ovlp': np.nanmean(precision_hp),
                'recall_ovlp': np.nanmean(recall_hp),
                #'avg_TP_SZ_overlap': np.nanmean(TP_SZ_overlap_hp),
                'avg_far': np.nanmean(far_hp),
                'avg_tiw': np.nanmean(tiw_hp),
                'avg_percentage_tiw': np.nanmean(percentage_tiw_hp)
            }
        )
    inner_cv_results = pl.DataFrame(inner_cv_results_rows)
    s.RESULTS_DIR.parent.mkdir(exist_ok=True, parents=True)
    inner_cv_results.write_csv(s.RESULTS_DIR / f'fold_{str(outer_fold_idx + 1)}_{clf}_grid_search_results.csv')
    
    # refit the model on the whole data using the best selected hyperparameter,
    # and return the fitted model
    best_hp = hyperparams[np.argmax(all_scores)]
    logger.info(f'Outer fold {outer_fold_idx+1} grid search finished')
    logger.info(f'\t ** Grid search: keep best hyperparameters combination = {best_hp} **')
    logger.info(f'\t ** Highest f1-score (regularized) from the grid search is {np.max(all_scores)}')
    best_model = clone(model)

    #X = data.iloc[:, 4:-1]
    X = data.select([pl.col(col) for col in feature_list])
    #y = data.iloc[:, -1]
    y = data.select('label')

    sc = StandardScaler()
    X = sc.fit_transform(X)

    if model.__class__.__name__ == 'SVC':
        params = {'kernel': best_hp[0], 'C': best_hp[1], 'class_weight': 'balanced'}
    elif model.__class__.__name__ == 'LogisticRegression':
        params = {'solver': best_hp[0], 'C': best_hp[1], 'class_weight': 'balanced'}
    elif model.__class__.__name__ == 'DecisionTreeClassifier':
        params = {'splitter': best_hp[0], 'criterion': best_hp[1], 'class_weight': 'balanced'}
    elif model.__class__.__name__ == 'KNeighborsClassifier':
        params = {'algorithm': best_hp[0], 'n_neighbors': best_hp[1]}
    elif model.__class__.__name__ == 'XGBClassifier':
        params = {'max_depth': best_hp[0], 'min_child_weight': best_hp[1],
                  'scale_pos_weight': 3000, 'max_delta_step': 1,
                  'learning_rate': 0.1, 'gamma': 0.1, 'booster': 'gbtree'}
    # TODO :find the scale pos weight for EEG datasets
    # NOTE: Since our dataset is imbalanced with 1h non-seizure data for approx. 2min of seizure data
    # we empirically set the XGBoost parameter scale_pos_weight to 3000 to control the balance of classes
    # Typical value to consider is sum(negative instances) / sum(positive instances)
    # Other parameters are set empirically to reduce overfitting and avoid underfiting
    # You could recalibrate scale_pos_weight in each fold

    else:
        params = {}
    best_model.set_params(**params)
    best_model.fit(X, y)

    return {'best_model': best_model, 'best_hyperparam': best_hp, 'scaler': sc}


def cross_validate(model, hyperparams:list, data:pl.DataFrame,
                   k:int, inner_k:int,
                   feature_group: str = 'all',
                   #home_path=home
                   ):
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

    if feature_group == "all":
        feature_list = [
            feature
            for feature_list in s.FEATURE_GROUPS.values()
            for feature in feature_list
        ]
    else:
        feature_list = s.FEATURE_GROUPS[feature_group]

    results_rows = []
    df_roc_curves = pl.DataFrame([])

    clf = model.__class__.__name__
    feature_list = []
    # nonfeature_list = ['unique_id', 'subject', 'session', 'run', 'dataset_name', 'second', 'timestamp' 'label', 'training']
    
    #win_size = hyperparams[0][2]
    #step = hyperparams[0][3]
    all_scores = []

    for i, (train_idx, test_idx) in enumerate(get_train_test_splits(data[['subject']], k)):
        
        logger.info(f"\nOuter CV loop: fold {i}")

        train_val_set = data[train_idx]
        test_set = data[test_idx]

        X_test = test_set.select([pl.col(col) for col in feature_list])
        # or exclude non-features columns
        #X_test = test_set.select([col for col in test_set.columns if col not in nonfeature_list])

        gridsearch_per_fold = grid_search(model, hyperparams, train_val_set, inner_k,
                                          outer_fold_idx=i,
                                          feature_list=feature_list
                                          #home_path=home_path
                                          )
        best_model = gridsearch_per_fold['best_model']
        best_hp = gridsearch_per_fold['best_hyperparam']
        scaler = gridsearch_per_fold['scaler']

        model_name = f'fold_{i}_{clf}_{best_hp[0]}_{best_hp[1]}.sav'
        pickle.dump(best_model, open(s.RESULTS_DIR / model_name, 'wb'))

        X_test = scaler.transform(X_test)
        
        # Make predictions and calculate performance metrics
        metrics = calculate_metrics(best_model, test_set, scaled_X_test=X_test,
                                    tau=best_hp[4], threshold=best_hp[5], step=best_hp[3],
                                    show=False)
        
        results_rows.append(
            {
            'fold': i,
            'train_subject': train_val_set.select(pl.col("subject").unique()).to_series().to_list(),
            'test_subject': test_set.select(pl.col("subject").unique()).to_series().to_list(),
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
                'fold': np.ones(len(metrics['roc_fpr'])) * (i),
                'roc_fpr': metrics['roc_fpr'],
                'roc_tpr': metrics['roc_tpr'],
                'roc_thresholds': metrics['roc_thresholds']
            }
        )

        logger.info(f"Outer CV loop: finished fold {i}, f1-score={metrics['ovlp_f1']}%, ROC-AUC={metrics['roc_auc_score']}")
        all_scores.append(metrics['ovlp_f1'])
        df_roc_curves = pl.concat([df_roc_curves, df_roc], how="vertical")

    df_results = pl.DataFrame(results_rows)

    s.RESULTS_DIR.parent.mkdir(exist_ok=True, parents=True)
    df_results.write_csv(s.RESULTS_DIR / "cv_results.csv")
    df_roc_curves.write_csv(s.RESULTS_DIR / "cv_roc_curves.csv")
    print("Results have been successfully stored.")

    return all_scores