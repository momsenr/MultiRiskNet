import cancerrisknet.datasets.factory as dataset_factory
from cancerrisknet.utils.time_logger import TimeLogger
import os
from os.path import dirname, realpath
import sys
import json
from cancerrisknet.utils.parsing import parse_args, parse_dispatcher_config
from cancerrisknet.datasets.disease_progression import DiseaseProgressionDataset
import cancerrisknet.models.factory as model_factory
import cancerrisknet.learn.train_debug as train
import cancerrisknet.utils.eval as eval
import datetime
import orjson

from cancerrisknet.utils.learn import init_metrics_dictionary, \
    get_dataset_loader, get_train_variables

if __name__ == '__main__':
    config_dir = "configs"
    config_file = "PC_MarketS.json"
    config_path = os.path.join(config_dir, config_file)
    config = json.load(open(config_path, 'r'))
    flags = parse_dispatcher_config(config)
    print(flags)
    # args = parse_args(flags[0]+' --save_dir test  --resume_from_result  results/PC_MS_1948_8fb3b279_20230615-1701/8fb3b279193046051296c42d16192d62.results')
    args = parse_args(flags[
                          0] + ' --save_dir test  --resume_from_result results/PC_MS_1951_47892d43_20230616-2253/47892d43d4ded404cda30c612a8d963a.results')

    # logger_main = TimeLogger(args, 1, hierachy=5, model_name=args.results_path) if args.time_logger_verbose >= 1 else TimeLogger(args, 0, model_name=args.results_path)
    # logger_main.log("Now main.py starts...")
    data_dir = 'data'
    metadata_path = os.path.join(data_dir, 'MS_full_0622_anon.json')
    print("Loading Dataset...")
    print(datetime.datetime.now())
    metadata = json.load(open(metadata_path, 'r'))
    print(datetime.datetime.now())
    del metadata
    print(datetime.datetime.now())
    metadata = orjson.loads(open(metadata_path, 'r').read)
    print(datetime.datetime.now())

    args.code_to_index_map = json.load(open(args.results_path + '.code_map', 'r'))
    args.index_map_length = len(args.code_to_index_map)
    # dataset_class = get_dataset_class(args)

    train_data = DiseaseProgressionDataset(metadata, args, 'train')

    print("Loading model...")
    model = model_factory.load_model(args.snapshot, args)
    data_loader = get_dataset_loader(args, train_data)

    # Set up models
    if isinstance(model, dict):
        models = model
    else:
        models = {args.model_name: model}

    loss, golds, gold_seqs, patient_golds, preds, probs, pids, censor_times, days_to_final_censors, dates = train.run_epoch(
        data_loader,
        train=False,
        truncate_epoch=False,  # (not args.exhaust_dataloader and eval_data.split_group != 'test'),
        models=models,
        optimizers=None,
        args=args
    )
    eval_stats = init_metrics_dictionary()
    name="transformer"

    log_statement, eval_stats, eval_preds = eval.compute_eval_metrics(
            args, loss,
            golds, patient_golds, probs, pids, dates,
            censor_times, days_to_final_censors, eval_stats, name)
    print(log_statement)

    # Todo: find out what patient_golds are and understand why they are nondeterministic!
    # print(patient_golds)

    # Todo: Also, the number of trajectories in the minibatch seems to be the number of patients,
    # however, there is still a random element to it - maybe due to balancing?????

    print('golds')
    print(golds)
    print("gold_seqs")
    print(gold_seqs)

    print("probs")
    print(probs)

    print(exams)
    print("pids")
    print(pids)

    print(
        "the biggest question remains: Why did the model learn probabilities which are the same for all 6 timepoints???")
    print("probably because there are not enough long trajectories in the model")
    # train_data, dev_data, test_data, attribution_set, args = dataset_factory.get_dataset(args)
    # print ("Number of patient for -Train:{},-Dev:{}, -Test:{}, --Attr:{}".format(
    #    train_data.__len__(), dev_data.__len__(), test_data.__len__(), attribution_set.__len__()))
    # logger_main.log("Load datasets")

