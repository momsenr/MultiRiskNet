import sklearn.metrics
from cancerrisknet.utils.c_index import concordance_index
import warnings
import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve


def get_probs_golds(test_preds, index=4, cancer_class=True):
    """
    Get pairs of predictions and labels that passed the data pre-processing criteria.

    Args:
        test_preds:
        index: the position at which the prediction vector (default: [3,6,12,36,60]) is evaluated.

    Returns:
        A pair of lists with the same length, ready for the use of AUROC, AUPRC, and etc.

    """

    probs_for_eval = [prob_arr[index] for prob_arr in test_preds["probs"]]
    golds_for_eval = [gold_arr[index] for gold_arr in test_preds["golds"]]

    return probs_for_eval, golds_for_eval


def compute_eval_metrics(args, loss, golds, probs, pids, dates, censor_time_indices,
                         days_to_final_censors, stats_dict, key_prefix):
    
    stats_dict['{}_loss'.format(key_prefix)].append(loss)
    preds_dict = {
        'golds': golds,
        'probs': probs,
        'pids': pids,
        'dates': dates,
        'censor_time_indices': censor_time_indices,
        'days_to_final_censors': days_to_final_censors
    }

    log_statement = '-- loss: {:.6f}'.format(loss)

    sum_auprc=0
    sum_auroc=0
    sum_mcc=0

    for index, time in enumerate(args.month_endpoints):
        probs_for_eval = [prob_arr[index] for prob_arr in probs]
        golds_for_eval = [gold_arr[index] for gold_arr in golds]

        if args.eval_auroc:
            key_name = '{}_{}month_auroc'.format(key_prefix, time)
            auc = compute_auroc(golds_for_eval, probs_for_eval)
            log_statement += " -{}: {} (n={} , c={} )".format(key_name, auc, len(golds_for_eval), sum(golds_for_eval))
            stats_dict[key_name].append(auc)
            if(auc!='NA'):
                sum_auroc+=auc

        if args.eval_auprc:
            key_name = '{}_{}month_auprc'.format(key_prefix, time)
            auc = compute_auprc(golds_for_eval, probs_for_eval)
            log_statement += " -{}: {} (n={} , c={} )".format(key_name, auc, len(golds_for_eval), sum(golds_for_eval))    
            stats_dict[key_name].append(auc)
            if(auc!='NA'):
                sum_auprc+=auc

        if args.eval_mcc:
            key_name = '{}_{}month_mcc'.format(key_prefix, time)
            mcc = compute_mcc(golds_for_eval, probs_for_eval)
            log_statement += " -{}: {} (n={} , c={} )".format(key_name, mcc, len(golds_for_eval), sum(golds_for_eval))
            stats_dict[key_name].append(mcc)
            if(auc!='NA'):
                sum_mcc+=mcc

    if args.eval_auroc:
        key_name_auroc_sum = '{}_sum_auroc'.format(key_prefix)
        log_statement += " -{}: {} ".format(key_name_auroc_sum, sum_auroc)
        stats_dict[key_name_auroc_sum].append(sum_auroc)
    if args.eval_auprc:
        key_name_auprc_sum = '{}_sum_auprc'.format(key_prefix)
        log_statement += " -{}: {} ".format(key_name_auprc_sum, sum_auprc)   
        stats_dict[key_name_auprc_sum].append(sum_auprc)
    if args.eval_mcc:
        key_name_mcc_sum = '{}_sum_mcc'.format(key_prefix)
        log_statement += " -{}: {} ".format(key_name_mcc_sum, sum_mcc)   
        stats_dict[key_name_mcc_sum].append(sum_mcc)

    if args.eval_c_index:
        c_index = compute_c_index(probs, censor_time_indices, golds)
        stats_dict['{}_c_index'.format(key_prefix)].append(c_index)
        log_statement += " -c_index: {}".format(c_index)
    

    return log_statement, stats_dict, preds_dict

def compute_eval_metrics_multiclass(args, loss, golds, probs, pids, dates, censor_time_indices, days_to_final_censors
                                    , stats_dict, key_prefix):
    all_log_statements = []
    all_stats_dicts = []
    all_preds_dicts = []
    
    # Convert lists to NumPy arrays
    golds_array = np.array(golds)
    probs_array = np.array(probs)
    censor_time_indices_array=np.array(censor_time_indices)

    #print debug statements
    print("example golds_array: ", golds_array[0])
    print("example probs_array: ", probs_array[0])
    print("example censor_time_indices_array: ", censor_time_indices_array[0])

    for task_idx in range(args.num_tasks+1):
        #Here we evaluate the performance of the model on each task/class vs. the rest of the classes
        task_key_prefix = f"{key_prefix}_task{task_idx}"
        class_idx=task_idx

        # Create binary arrays for true labels and predictions
        binary_golds = (golds_array == class_idx).astype(int)
        binary_predictions = probs_array[:, class_idx]

        log_statement, task_stats_dict, task_preds_dict = compute_eval_metrics(
            args, loss, binary_golds, binary_predictions, pids, dates, censor_time_indices_array,
            days_to_final_censors, stats_dict, task_key_prefix
        )

        all_log_statements.append(log_statement)
        all_stats_dicts.append(task_stats_dict)
        all_preds_dicts.append(task_preds_dict)

    # Combine all stats_dicts and preds_dicts
    for d in all_stats_dicts:
        stats_dict.update(d)

    sum_auroc=0
    sum_auprc=0
    sum_mcc=0
    for task_idx in range(args.num_tasks):
        task_key_prefix = f"{key_prefix}_task{task_idx}" 
        if args.eval_auroc:
            key=task_key_prefix+"_sum_auroc"
            sum_auroc+=stats_dict[key][-1]
        if args.eval_auprc:
            key=task_key_prefix+"_sum_auprc"
            sum_auprc+=stats_dict[key][-1]
        if args.eval_mcc:
            key=task_key_prefix+"_sum_mcc"
            sum_mcc+=stats_dict[key][-1]
    
    stats_dict['{}_loss'.format(key_prefix)].append(loss)

    if args.eval_auroc:
        key_name_auroc_sum = '{}_all_tasks_sum_auroc'.format(key_prefix)
        all_log_statements.append(" -{}: {} ".format(key_name_auroc_sum, sum_auroc))
        stats_dict[key_name_auroc_sum].append(sum_auroc)
    if args.eval_auprc:
        key_name_auprc_sum = '{}_all_tasks_sum_auprc'.format(key_prefix)
        all_log_statements.append(" -{}: {} ".format(key_name_auprc_sum, sum_auprc))
        stats_dict[key_name_auprc_sum].append(sum_auprc)
    if args.eval_mcc:
        key_name_mcc_sum = '{}_all_tasks_sum_mcc'.format(key_prefix)
        all_log_statements.append(" -{}: {} ".format(key_name_mcc_sum, sum_mcc))
        stats_dict[key_name_mcc_sum].append(sum_mcc)
    
    combined_preds_dict = {key: [d[key] for d in all_preds_dicts] for key in all_preds_dicts[0]}
    
    # Combine all task log statements into one
    combined_log_statement = "\n".join(all_log_statements)

    return combined_log_statement, stats_dict, combined_preds_dict


def include_exam_and_determine_label(current_eval_index, censor_time_index, gold):
    """
        Determine if a given prediction should be evaluated in this pass. Evalute for the time interval 
        *up to a given time point*, e.g. there is (not) a cancer dianosis until the 36 months after
        time of assessment.

    Args:
        current_eval_index: the index of the prediction that is being evaluated
        censor_time_index: the position at which the outcome is censored (i.e. cancer or death happens)
        gold: the ground truth (whether this trajectory is associated with a cancer dianosis or not.
    """

    valid_pos = (gold.sum()!=0) and censor_time_index <= current_eval_index
    valid_neg = censor_time_index >= current_eval_index
    included, label = (valid_pos or valid_neg), valid_pos
    return included, label
    

def compute_c_index(probs, censor_time_indices, golds):
    try:
        c_index = concordance_index(censor_time_indices, probs, golds)
    except Exception as e:
        warnings.warn("Failed to calculate C-index because {}".format(e))
        c_index = 'NA'
    return c_index


def compute_auroc(golds_for_eval, probs_for_eval):
    try:
        fpr, tpr, _ = sklearn.metrics.roc_curve(golds_for_eval, probs_for_eval, pos_label=1)
        auc = sklearn.metrics.roc_auc_score(golds_for_eval, probs_for_eval, average='samples')
    except Exception as e:
        warnings.warn("Failed to calculate AUROC because {}".format(e))
        auc = 'NA'
    return auc


def compute_auprc(golds_for_eval, probs_for_eval):
    try:
        precisions, recalls, _ = sklearn.metrics.precision_recall_curve(golds_for_eval, probs_for_eval, pos_label=1)
        auc = sklearn.metrics.auc(recalls, precisions)
    except Exception as e:
        warnings.warn("Failed to calculate AUPRC because {}".format(e))
        auc = 'NA'
    return auc


def compute_mcc(golds_for_eval, probs_for_eval):
    try:
        p = sum(golds_for_eval)
        n = sum([not el for el in golds_for_eval])
        fp, tp, thresholds = _binary_clf_curve(golds_for_eval, probs_for_eval)
        tn, fn = n - fp, p - tp
        mcc = (tp * tn - fp * fn) / (np.sqrt(((tp + fp) * (fp + tn) * (tn + fn) * (fn + tp))) + 1e-10)
    except Exception as e:
        warnings.warn("Failed to calculate MCC because {}".format(e))
        mcc = 'NA'
    return max(mcc)
