import json
import tqdm
from collections import Counter, defaultdict
import pandas as pd
import os
import sys
from os.path import dirname, realpath
sys.path.insert(0, dirname(dirname(realpath(__file__))))
from cancerrisknet.utils.parsing import md5
from cancerrisknet.utils.parsing import get_code
import pickle
import pandas as pd


NO_DATASET_ERR = "Dataset {} not in DATASET_REGISTRY! Available datasets are {}"
DATASET_REGISTRY = {}
NUM_PICKLES = 50
UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'


def RegisterDataset(dataset_name):
    """Registers a dataset. Used as a decorator."""

    def decorator(f):
        DATASET_REGISTRY[dataset_name] = f
        return f

    return decorator


def get_dataset_class(args):
    if args.dataset not in DATASET_REGISTRY:
        raise Exception(
            NO_DATASET_ERR.format(args.dataset, DATASET_REGISTRY.keys()))

    return DATASET_REGISTRY[args.dataset]


def build_code_to_index_map(args):
    """
        Create a mapping dict for each of the categorical token (diagnosis code) occured in the dataset.
        Require input file `data/all_observed_icd.txt` which should be automatically generated during data pre-processing
        following steps under `scripts/metadata/`.
    """
    print("Building code to index map...")

    if(args.metadata_path.endswith('/')):
        # Split the string from the right side by '/'
        parts = args.metadata_path.rsplit('/', 1)

        # Join the parts back together with '-vocab.txt' in place of the last '/'
        new_path = parts[0] + '-vocab.txt'
        vocab_path = os.path.join(
            os.path.dirname(parts[0]), os.path.basename(new_path)
        )
    else:
        vocab_path = os.path.join(
            os.path.dirname(args.metadata_path), os.path.basename(args.metadata_path).replace('.h5', '-vocab.txt')
        )

    with open(vocab_path, 'r') as f:
        # Read df from file
        pd_df = pd.read_csv(f, header=None)
        # Convert df to list
        all_codes = pd_df[0].tolist()

    #2024-01-15 we don't need to run any preprocessing here anymore
    all_observed_codes = all_codes#[get_code(args, code) for code in all_codes]
    print("Length of all_observed", len(all_observed_codes))
    all_codes_counts = dict(Counter(all_observed_codes))
    all_codes = list(all_codes_counts.keys())
    all_codes_p = list(all_codes_counts.values())
    all_codes_p = [i/sum(all_codes_p) for i in all_codes_p]
    code_to_index_map = {code: i+1 for i, code in enumerate(all_codes)}
    code_to_index_map.update({
        PAD_TOKEN: 0,
        UNK_TOKEN: len(code_to_index_map)+1
        })
    args.code_to_index_map = code_to_index_map
    args.all_codes = all_codes
    args.all_codes_p = all_codes_p


def get_dataset(args):
    """
        Generate torch-compatible dataset instances for training, evaluation or any other analysis.
    """
    # Depending on arg, build dataset
    #if (not args.metadata_path.endswith('.h5')):
    #    raise Exception("Metadata file must be in hdf5 format")

    dataset_class = get_dataset_class(args)

    if(args.data_is_preprocessed==False):
        preprocess_train=True
        preprocess_dev=True
        preprocess_test=True
    else:
        preprocess_train=False
        preprocess_dev=False
        preprocess_test=False

    train = dataset_class(args, 'train',args.metadata_path, preprocess_train) if args.train else []
    dev = dataset_class(args, 'dev',args.metadata_path, preprocess_dev) if args.train or args. dev else []
    test = dataset_class(args, 'test',args.metadata_path, preprocess_test) if args.test else []

    if args.attribute:
        attr = dataset_class(args, 'test',args.metadata_path, False)
        attr.split_group='attribute'
    else:
        attr = []

    # Build a new code to index map (previously this was done only during training)
    build_code_to_index_map(args)
    json.dump(args.code_to_index_map, open(args.results_path + '.code_map', 'w'))

    args.index_map_length = len(args.code_to_index_map)

    if args.max_events_length is None:
        args.max_events_length = max([len(record['codes']) for record in train.dataset])

    if args.pad_size is None:
        args.pad_size = args.max_events_length

    args.PAD_TOKEN = PAD_TOKEN
    return train, dev, test, attr, args
