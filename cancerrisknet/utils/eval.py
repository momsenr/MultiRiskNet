import sklearn.metrics
from cancerrisknet.utils.c_index import concordance_index
import warnings
import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve


def get_probs_golds(test_preds, index=4):
    """
    Get pairs of predictions and labels that passed the data pre-processing criteria.

    Args:
        test_preds:
        index: the position at which the prediction vector (default: [3,6,12,36,60]) is evaluated.

    Returns:
        A pair of lists with the same length, ready for the use of AUROC, AUPRC, and etc.

    """

    probs_for_eval, golds_for_eval = [], []
    for prob_arr, censor_time_index, gold in zip(test_preds["probs"], test_preds["censor_time_indices"], test_preds["golds"]):
        include, label = include_exam_and_determine_label(index, censor_time_index, gold)
        if include:
            probs_for_eval.append(prob_arr[index])
            golds_for_eval.append(label)

    return probs_for_eval, golds_for_eval


def compute_eval_metrics(args, loss, golds, patient_golds, probs, pids, dates, censor_time_indices,
                         days_to_final_censors, stats_dict, key_prefix):
    
    stats_dict['{}_loss'.format(key_prefix)].append(loss)
    preds_dict = {
        'golds': golds,
        'probs': probs,
        'patient_golds': patient_golds,
        'pids': pids,
        'dates': dates,
        'censor_time_indices': censor_time_indices,
        'days_to_final_censors': days_to_final_censors
    }

    log_statement = '-- loss: {:.6f}'.format(loss)

    sum_auprc=0
    sum_auroc=0
    sum_mcc=0
    weighed_sum_auprc=0
    weighed_sum_auroc=0
    weighed_sum_mcc=0

    for index, time in enumerate(args.month_endpoints):
        probs_for_eval, golds_for_eval = get_probs_golds(preds_dict, index=index)

        if args.eval_auroc:
            key_name = '{}_{}month_auroc'.format(key_prefix, time)
            auc = compute_auroc(golds_for_eval, probs_for_eval)
            log_statement += " -{}: {} (n={} , c={} )".format(key_name, auc, len(golds_for_eval), sum(golds_for_eval))
            stats_dict[key_name].append(auc)
            sum_auroc+=auc
            weighed_sum_auroc+=auc*(sum(golds_for_eval)/len(golds_for_eval))

        if args.eval_auprc:
            key_name = '{}_{}month_auprc'.format(key_prefix, time)
            auc = compute_auprc(golds_for_eval, probs_for_eval)
            log_statement += " -{}: {} (n={} , c={} )".format(key_name, auc, len(golds_for_eval), sum(golds_for_eval))    
            stats_dict[key_name].append(auc)
            sum_auprc+=auc
            weighed_sum_auprc+=auc*(sum(golds_for_eval)/len(golds_for_eval))

        if args.eval_mcc:
            key_name = '{}_{}month_mcc'.format(key_prefix, time)
            mcc = compute_mcc(golds_for_eval, probs_for_eval)
            log_statement += " -{}: {} (n={} , c={} )".format(key_name, mcc, len(golds_for_eval), sum(golds_for_eval))
            stats_dict[key_name].append(mcc)
            sum_mcc+=mcc
            weighed_sum_mcc+=mcc*(sum(golds_for_eval)/len(golds_for_eval))

    if args.eval_auroc:
        key_name_auroc_sum = '{}_sum_auroc'.format(key_prefix)
        key_name_weighed_auroc_sum = '{}_weighed_sum_auroc'.format(key_prefix)
        log_statement += " -{}: {} ".format(key_name_auroc_sum, sum_auroc)
        log_statement += " -{}: {} ".format(key_name_weighed_auroc_sum, weighed_sum_auroc)
        stats_dict[key_name_auroc_sum].append(sum_auroc)
        stats_dict[key_name_weighed_auroc_sum].append(weighed_sum_auroc)
    if args.eval_auprc:
        key_name_auprc_sum = '{}_sum_auprc'.format(key_prefix)
        key_name_weighed_auprc_sum = '{}_weighed_sum_auprc'.format(key_prefix)
        log_statement += " -{}: {} ".format(key_name_auprc_sum, sum_auprc)   
        log_statement += " -{}: {} ".format(key_name_weighed_auprc_sum, weighed_sum_auprc)
        stats_dict[key_name_auprc_sum].append(sum_auprc)
        stats_dict[key_name_weighed_auprc_sum].append(weighed_sum_auprc)
    if args.eval_mcc:
        key_name_mcc_sum = '{}_sum_mcc'.format(key_prefix)
        key_name_weighed_mcc_sum = '{}_weighed_sum_mcc'.format(key_prefix)
        log_statement += " -{}: {} ".format(key_name_mcc_sum, sum_mcc)   
        log_statement += " -{}: {} ".format(key_name_weighed_mcc_sum, weighed_sum_mcc)
        stats_dict[key_name_mcc_sum].append(sum_mcc)
        stats_dict[key_name_weighed_mcc_sum].append(weighed_sum_mcc)

    if args.eval_c_index:
        c_index = compute_c_index(probs, censor_time_indices, golds)
        stats_dict['{}_c_index'.format(key_prefix)].append(c_index)
        log_statement += " -c_index: {}".format(c_index)
    

    return log_statement, stats_dict, preds_dict


def compute_eval_metrics_multitask(args, loss, golds, patient_golds, probs, pids, dates, censor_time_indices, days_to_final_censors
                                    , stats_dict, key_prefix):
    all_log_statements = []
    all_stats_dicts = []
    all_preds_dicts = []

    for task_idx in range(args.num_tasks):
        task_key_prefix = f"{key_prefix}_task{task_idx}"

        # Extract metrics for the current task
        task_golds = np.array([arr[task_idx] for arr in golds])
        task_patient_golds = np.array([arr[task_idx] for arr in patient_golds]) 
        task_probs =  np.array([arr[task_idx] for arr in probs]) 
        task_censor_time_indices = np.array([arr[task_idx] for arr in censor_time_indices])
        


        log_statement, task_stats_dict, task_preds_dict = compute_eval_metrics(
            args, loss, task_golds, task_patient_golds, task_probs, pids, dates, task_censor_time_indices,
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
    weighed_sum_auroc=0
    weighed_sum_auprc=0
    weighed_sum_mcc=0
    for task_idx in range(args.num_tasks):
        task_key_prefix = f"{key_prefix}_task{task_idx}" 
        if args.eval_auroc:
            key=task_key_prefix+"_sum_auroc"
            weighed_key=task_key_prefix+"_weighed_sum_auroc"
            sum_auroc+=stats_dict[key][-1]
            weighed_sum_auroc+=stats_dict[weighed_key][-1]
        if args.eval_auprc:
            key=task_key_prefix+"_sum_auprc"
            weighed_key=task_key_prefix+"_weighed_sum_auprc"
            sum_auprc+=stats_dict[key][-1]
            weighed_sum_auprc+=stats_dict[weighed_key][-1]
        if args.eval_mcc:
            key=task_key_prefix+"_sum_mcc"
            weighed_key=task_key_prefix+"_weighed_sum_mcc"
            sum_mcc+=stats_dict[key][-1]
            weighed_sum_mcc+=stats_dict[weighed_key][-1]
    
    if args.eval_auroc:
        key_name_auroc_sum = '{}_all_tasks_sum_auroc'.format(key_prefix)
        key_name_weighed_sum_auroc = '{}_all_tasks_weighed_sum_auroc'.format(key_prefix)
        all_log_statements.append(" -{}: {} ".format(key_name_auroc_sum, sum_auroc))
        all_log_statements.append(" -{}: {} ".format(key_name_weighed_sum_auroc, weighed_sum_auroc))
        stats_dict[key_name_auroc_sum].append(sum_auroc)
        stats_dict[key_name_weighed_sum_auroc].append(weighed_sum_auroc)
    if args.eval_auprc:
        key_name_auprc_sum = '{}_all_tasks_sum_auprc'.format(key_prefix)
        key_name_weighed_sum_auprc = '{}_all_tasks_weighed_sum_auprc'.format(key_prefix)
        all_log_statements.append(" -{}: {} ".format(key_name_auprc_sum, sum_auprc))
        all_log_statements.append(" -{}: {} ".format(key_name_weighed_sum_auprc, weighed_sum_auprc))
        stats_dict[key_name_auprc_sum].append(sum_auprc)
        stats_dict[key_name_weighed_sum_auprc].append(weighed_sum_auprc)
    if args.eval_mcc:
        key_name_mcc_sum = '{}_all_tasks_sum_mcc'.format(key_prefix)
        key_name_weighed_sum_mcc = '{}_all_tasks_weighed_sum_mcc'.format(key_prefix)
        all_log_statements.append(" -{}: {} ".format(key_name_mcc_sum, sum_mcc))
        all_log_statements.append(" -{}: {} ".format(key_name_weighed_sum_mcc, weighed_sum_mcc))
        stats_dict[key_name_mcc_sum].append(sum_mcc)
        stats_dict[key_name_weighed_sum_mcc].append(weighed_sum_mcc)
    
    combined_preds_dict = {key: [d[key] for d in all_preds_dicts] for key in all_preds_dicts[0]}
    
    # Combine all task log statements into one
    combined_log_statement = "\n".join(all_log_statements)

    return combined_log_statement, stats_dict, combined_preds_dict


def include_exam_and_determine_label(followup, censor_time_index, gold, cumulative_prediction_interval=True):
    """
        Determine if a given prediction should be evaluated in this pass.

    Args:
        followup:
        censor_time_index: the position at which the prediction vector (default: [3,6,12,36,60]) is evaluated.
        gold: the ground truth (whether this trajectory is associated with a cancer dianosis or not.
        cumulative_prediction_interval: One of ['c', 'i'].
                                        If 'c' then evalute for the time interval *up to a given time point*,
                                            e.g. there is (not) a cancer dianosis until the 36 months after
                                                 time of assessment.
                                        If 'i' then evalute for the exact time interval for a given time point,
                                            e.g. there is (not) a cancer dianosis occurrence between 12-36 months after
                                                 time of assessment.
    """
    if cumulative_prediction_interval:
        valid_pos = gold and censor_time_index <= followup
    else:
        valid_pos = gold and censor_time_index == followup
    valid_neg = censor_time_index >= followup
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
