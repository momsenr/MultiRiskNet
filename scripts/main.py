import os
from os.path import dirname, realpath
import sys
from glob import glob
sys.path.insert(0, dirname(dirname(realpath(__file__))))
import pickle
import cancerrisknet.datasets.factory as dataset_factory
import cancerrisknet.models.factory as model_factory
from cancerrisknet.models.utils import AttributionModel
import cancerrisknet.learn.train as train
import cancerrisknet.learn.attribute as attribute
from cancerrisknet.utils.parsing import parse_args
from cancerrisknet.utils.time_logger import TimeLogger
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    logger_main = TimeLogger(args, 1, hierachy=5, model_name=args.results_path) if args.time_logger_verbose >= 1 else TimeLogger(args, 0, model_name=args.results_path)
    logger_main.log("Now main.py starts...")
    print("CUDA:", torch.cuda.is_available())

    print("Loading Dataset...")
    train_data, dev_data, test_data, attribution_set, args = dataset_factory.get_dataset(args)
    print ("Number of patient for -Train:{},-Dev:{}, -Test:{}, --Attr:{}".format(
        train_data.__len__(), dev_data.__len__(), test_data.__len__(), attribution_set.__len__()))
    logger_main.log("Load datasets")

    if args.resume_from_result is None:
        print("Building model...")
        model = model_factory.get_model(args)
        results_path = args.results_path
    else:
        print("Loading model...")
        model = model_factory.load_model(args.snapshot, args)
        save_idx = 1 + len(glob(args.results_path + '-[0-9]'))
        results_path = args.results_path + '-{}'.format(save_idx)

    print(model)
    print("Working threads: ", torch.get_num_threads())
    if torch.get_num_threads() < args.num_workers:
        torch.set_num_threads(args.num_workers)
        print("Adding threads count to {}.".format(torch.get_num_threads()))

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in ['optimizer_state', 'code_to_index_map', 'all_codes', 'all_codes_p']:
            print("\t{}={}".format(attr.upper(), value))
    logger_main.log("Build model")

    print()
    if args.train:
        epoch_stats, model = train.train_model(train_data, dev_data, model, args)
        print("Save train/dev results to {}".format(args.results_path))
        logger_main.log("TRAINING")
        args_dict = vars(args).copy(); del args_dict['code_to_index_map']; pickle.dump(args_dict, open(results_path, 'wb'))
        pickle.dump(epoch_stats, open("{}.{}".format(args.results_path, "epoch_stats"), 'wb'))
        train_loss=epoch_stats['train_loss']
        dev_loss=epoch_stats['dev_loss']
        del epoch_stats
        logger_main.log("Dump results")

        #Plot the train loss curve
        plt.plot(train_loss, label='Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training loss')
        plt.savefig(args.results_path+'train_loss.png')

        #Plot the dev loss curve
        plt.plot(dev_loss, label='Dev loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training loss')
        plt.savefig(args.results_path+'dev_loss.png')
    print()
    if args.dev:
        print("-------------\nDev")
        dev_stats, dev_preds = train.eval_model(dev_data, 'dev', model, args)
        print("Save dev results to {}".format(args.results_path))
        logger_main.log("VALIDATION")
        args_dict = vars(args).copy(); del args_dict['code_to_index_map']; pickle.dump(args_dict, open(results_path, 'wb'))
        pickle.dump(dev_stats, open("{}.{}".format(args.results_path, "dev_stats"), 'wb'))
        pickle.dump(dev_preds, open("{}.{}".format(args.results_path, "dev_preds"), 'wb'))
        del dev_stats, dev_preds
        logger_main.log("Dump results")

    print()
    if args.test:
        print("-------------\nTest")
        test_stats, test_preds = train.eval_model(test_data, 'test', model, args)
        print("Save test results to {}".format(args.results_path))
        args_dict = vars(args).copy(); del args_dict['code_to_index_map']; pickle.dump(args_dict, open(results_path, 'wb'))
        logger_main.log("TESTING")

        pickle.dump(test_stats, open("{}.{}".format(args.results_path, "test_stats"), 'wb'))
        pickle.dump(test_preds, open("{}.{}".format(args.results_path, "test_preds"), 'wb'))
        del test_stats, test_preds
        logger_main.log("Dump results")

    print()
    if args.attribute:
        print("-------------\nAttribution")
        model_for_attribution = AttributionModel(model, args)
        for class_index in range(1): #range(args.num_tasks):
            test_attribution, test_attribution_y, test_censored_attribution = attribute.compute_attribution(attribution_set, model_for_attribution, args, class_index=class_index, \
            model_for_preds=model, attribution_method=args.attribution_method, \
            pred_threshold=args.attribution_prediction_threshold, attribution_sum_method=args.attribution_sum_method)
            print("Save attribution results for task {} to {}".format(class_index,args.results_path))

            pickle.dump(test_attribution_y, open("{}.test_attribution_y_task{}".format(args.results_path,str(class_index)), 'wb'))
            pickle.dump(test_attribution, open("{}.test_attribution_task{}".format(args.results_path,str(class_index)), 'wb'))
            pickle.dump(test_censored_attribution, open("{}.test_censored_attribution_task{}".format(args.results_path,str(class_index)), 'wb'))
        args_dict = vars(args).copy(); del args_dict['code_to_index_map']; pickle.dump(args_dict, open(args.results_path, 'wb'))
        logger_main.log("ATTRIBUTION")
        logger_main.log("Dump results")
