import os
import numpy as np
import torch
from tqdm import tqdm
from cancerrisknet.learn.step import model_step
from cancerrisknet.utils.eval import compute_eval_metrics_multiclass
from cancerrisknet.utils.learn import init_metrics_dictionary, \
    get_dataset_loader, get_train_variables
from cancerrisknet.utils.time_logger import TimeLogger
import warnings
tqdm.monitor_interval = 0

def train_model(train_data, dev_data, model, args):
    """
        Train model and tune on dev set using args.tuning_metric. If model doesn't improve dev performance within
        args.patience epochs, then update the learning rate schedule (such as early stopping or halve the learning
        rate, and restore the model to the saved best and continue training. At the end of training, the function
        will restore the model to best dev version.

        Returns:
            epoch_stats: a dictionary of epoch level metrics for train and test
            returns models : dict of models, containing best performing model setting from this call to train
    """

    logger_epoch = TimeLogger(args, 1, hierachy=2) if args.time_logger_verbose >= 2 else TimeLogger(args, 0)

    start_epoch, epoch_stats, state_keeper, models, optimizers, tuning_key, num_epoch_sans_improvement = \
        get_train_variables(args, model)

    train_data_loader = get_dataset_loader(args, train_data)
    dev_data_loader = get_dataset_loader(args, dev_data)
    logger_epoch.log("Get train and dev dataset loaders")

    train_only_last_layers=False
    layers_frozen=False
    smart_loss=False

    for epoch in range(start_epoch, args.epochs + 1):

        print("-------------\nEpoch {}:".format(epoch))

        if(args.loss_weights=='smart' and epoch>args.epochs/4):
            print("Using smart loss")
            smart_loss=True
                    
        if(args.freeze_all_but_last_layer=='after_number_of_epochs' and epoch==args.freeze_all_but_last_layer_after_epoch):
            train_only_last_layers = True

        for mode, data_loader in [('Train', train_data_loader), ('Dev', dev_data_loader)]:
            if_train = mode == 'Train'
            key_prefix = mode.lower()

            if(train_only_last_layers and not layers_frozen):
                logger_epoch.log("Freezing all but last layers")
                layers_to_not_freeze = [
                    'prob_of_failure_layer.hazard_fcs.weight', 
                    'prob_of_failure_layer.hazard_fcs.bias',
                    'prob_of_failure_layer.base_hazard_fcs.weight', 
                    'prob_of_failure_layer.base_hazard_fcs.bias'
                ]

                for name, param in models[args.model_name].named_parameters():
                    for param_group in optimizers[args.model_name].param_groups:
                        if name not in layers_to_not_freeze:
                            param_group['lr'] = 0.0
                layers_frozen=True

            loss,  golds, probs, pids, censor_time_indices, days_to_final_censors, dates = \
                run_epoch(data_loader, train=if_train, truncate_epoch=True, models=models,
                          optimizers=optimizers, args=args, smart_loss=(smart_loss and if_train),smart_verbose=True)
            logger_epoch.log("Run epoch ({})".format(key_prefix))

            log_statement, epoch_stats, _ = compute_eval_metrics_multiclass(args, loss, golds, probs,
                                                                 pids, dates, censor_time_indices, days_to_final_censors,
                                                                 epoch_stats, key_prefix)
            logger_epoch.log("Compute eval metrics ({})".format(key_prefix))
            print(log_statement)

        # Save model if beats best dev (min loss or max c-index_{i,a})
        best_func, arg_best = (min, np.argmin) if 'loss' in tuning_key else (max, np.argmax)
        improved = best_func(epoch_stats[tuning_key]) == epoch_stats[tuning_key][-1]
        if improved:
            num_epoch_sans_improvement = 0
            os.makedirs(args.save_dir, exist_ok=True)
            epoch_stats['best_epoch'] = arg_best(epoch_stats[tuning_key])
            state_keeper.save(models, optimizers, epoch, args.lr, epoch_stats)
            logger_epoch.log("Save improved model")
        else:
            num_epoch_sans_improvement += 1

        print('---- Best Dev {} is {} at epoch {}'.format(
            args.tuning_metric,
            epoch_stats[tuning_key][epoch_stats['best_epoch']],
            epoch_stats['best_epoch'] + 1))

        if num_epoch_sans_improvement >= args.patience:
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0
            if(args.freeze_all_but_last_layer=='when_reducing_lr'):
                train_only_last_layers = True

            model_states, optimizer_states, _, _, _ = state_keeper.load()
            for name in models:
                model_state_dict = model_states[name]
                models[name].load_state_dict(model_state_dict)
            # Reset optimizers
            for name in optimizers:
                optimizer = optimizers[name]
                optimizer_state_dict = optimizer_states[name]
                optimizers[name].load_state_dict(optimizer_state_dict)
            # Reduce LR
            for name in optimizers:
                optimizer = optimizers[name]
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay

            # Update lr also in args for resumable usage
            args.lr *= args.lr_decay
            logger_epoch.log("Prepare for next epoch")
            logger_epoch.update()

    # Restore model to best dev performance, or last epoch when not tuning on dev
    model_states, _, _, _, _ = state_keeper.load()
    for name in models:
        model_state_dict = model_states[name]
        models[name].load_state_dict(model_state_dict)

    return epoch_stats, models

def run_epoch(data_loader, train, truncate_epoch, models, optimizers, args,smart_loss=False,smart_verbose=False):
    """
        Run model for one pass of data_loader, and return epoch statistics.
        Args:
            data_loader: Pytorch dataloader over some dataset.
            train: True to train the model and run the optimizers
            models: dict of models, where 'model' is the main model, and others can be critics, or meta-models
            truncate_epoch: used when dataset is too large, manually stop epoch after max_iteration without
                            necessarily spaning through the entire dataset.
            optimizers: dict of optimizers, one for each model
            args: general runtime args defined in by argparse

        Returns:
            avg_loss: epoch loss
            golds: labels for all samples in data_loader
            preds: model predictions for all samples in data_loader
            probs: model softmaxes for all samples in data_loader
            exams: exam ids for samples if available, used to cluster samples for evaluation.
    """
    data_iter = data_loader.__iter__()
    probs = []
    censor_time_indices = []
    days_to_final_censors = []
    dates = []
    golds = []
    losses = []
    pids = []
    logger = TimeLogger(args, args.time_logger_step) if args.time_logger_verbose >= 3 else TimeLogger(args, 0)

    torch.set_grad_enabled(train)
    for name in models:
        if train:
            models[name].train()
            if optimizers is not None:
                optimizers[name].zero_grad()
        else:
            models[name].eval()

    num_batches_per_epoch = len(data_loader)

    if truncate_epoch:
        max_batches = args.max_batches_per_train_epoch if train else args.max_batches_per_dev_epoch
        num_batches_per_epoch = min(len(data_loader), (max_batches))
        logger.log("Truncate epoch @ batches: {}".format(num_batches_per_epoch))
    i = 0
    tqdm_bar = tqdm(data_iter, total=num_batches_per_epoch)

    for batch in data_iter:
        if batch is None:
            warnings.warn('Empty batch')
            continue
        if tqdm_bar.n > num_batches_per_epoch:
            break
        
        with torch.no_grad():
            golds.extend(batch['y_seq'].data.numpy().astype(int))
            dates.extend(batch['admit_date'].data.numpy().astype(int))
            censor_time_indices.extend(batch['censor_time_index'].data.numpy().astype(int))
            days_to_final_censors.extend(batch['days_to_censor'].data.numpy().astype(int))
            pids.extend(batch['patient_id'].data.numpy().astype(int))

        batch = prepare_batch(batch, args)
        logger.newline()
        logger.log("prepare data")
        loss, batch_probs = model_step(batch, models, train, args, smart_loss=smart_loss,smart_verbose=smart_verbose)

        logger.log("model step")
        if train:
            optimizers[args.model_name].step()
            optimizers[args.model_name].zero_grad()

        logger.log("model update")
        with torch.no_grad():
            losses.append(loss)
            probs.extend(batch_probs)

        logger.log("saving results")

        i += 1
        if i > num_batches_per_epoch and args.num_workers > 0:
            data_iter.__del__()
            break
        logger.update()
        tqdm_bar.update()

    avg_loss = np.mean(losses)
    return avg_loss, golds, probs, pids, censor_time_indices, days_to_final_censors, dates


def prepare_batch(batch, args):
    to_gpu = ['x', 'time_seq', 'age', 'age_seq', 'y_mask']
    to_gpu_convert_to_float = ['y_seq']
    for key in batch.keys():
        if key in to_gpu:
            batch[key] = batch[key].to(args.device)
        elif key in to_gpu_convert_to_float:
            batch[key] = batch[key].int().to(args.device)
    return batch

def eval_model(eval_data, name, models, args):
    """
        Run model on test data, and return test stats (includes loss accuracy, etc)
    """
    logger_eval = TimeLogger(args, 1, hierachy=2) if args.time_logger_verbose >= 2 else TimeLogger(args, 0)
    logger_eval.log("Evaluating model")

    if not isinstance(models, dict):
        models = {args.model_name: models}
    models[args.model_name] = models[args.model_name].to(args.device)
    eval_stats = init_metrics_dictionary()
    logger_eval.log("Load model")

    data_loader = get_dataset_loader(args, eval_data)
    logger_eval.log('Load eval data')


    loss, golds, probs, pids, censor_time_indices, days_to_final_censors, dates = run_epoch(
        data_loader,
        train=False,
        truncate_epoch=(not args.exhaust_dataloader and eval_data.split_group != 'test'),
        models=models,
        optimizers=None,
        args=args
    )


    logger_eval.log('Run eval epoch')    


    log_statement, eval_stats, eval_preds = compute_eval_metrics_multiclass(
                            args, loss,
                            golds, probs, pids, dates,
                            censor_time_indices, days_to_final_censors, eval_stats, name)
    print(log_statement)
    logger_eval.log('Compute eval metrics')
    logger_eval.update()

    return eval_stats, eval_preds
