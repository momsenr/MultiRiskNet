from cancerrisknet.datasets.factory import RegisterDataset, UNK_TOKEN, PAD_TOKEN
from torch.utils import data
from cancerrisknet.utils.date import parse_date
from cancerrisknet.utils.parsing import get_code, md5, load_data_settings
import tqdm
from collections import Counter
import numpy as np
import random
import pandas as pd
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import os

MAX_TIME_EMBED_PERIOD_IN_DAYS = 120 * 365
MIN_TIME_EMBED_PERIOD_IN_DAYS = 10
SUMMARY_MSG = "Constructed disease progression {} dataset with {} records from {} patients, " \
              "and the following class balance:\n  {}"


@RegisterDataset("disease_progression")
class DiseaseProgressionDataset(data.Dataset):
    def __init__(self, args, split_group, path_to_data_parquet, preprocess_data=False):
        """
            Dataset for survival analysis based on categorical disease history information.

        Args:
            metadata (dict): The input metadata file (usually json) after pre-processing.
                             See `./data/README.md` for more details.
            split_group (str): Use any of ['train', 'test', 'dev'] or ['all', 'attribute'] for special usage.

        Returns:
            torch.utils.data.Dataset

        """
        super(DiseaseProgressionDataset, self).__init__()
        self.args = args
        self.split_group = split_group
        self.PAD_TOKEN = PAD_TOKEN
        self.path_to_data_parquet= path_to_data_parquet
        self.SETTINGS = load_data_settings(args)['SETTINGS']
        self.CANCER_CODE_dict = {"PC": '157 C25',
                            "OC": '719 C56'}
        self.num_tasks = args.num_tasks
        self.num_time_steps= len(self.args.month_endpoints)

        if(preprocess_data==True):
            print('not implemented in the DataLoader currently...')
            sys.exit(-1)
        else:
            print("Loading {} data from hard disk...".format(self.split_group))
            self.events=pq.read_table(self.path_to_data_parquet[:-1]+'_processed/split_group=' + self.split_group + '/').to_pandas()
            self.patients_with_valid_trajectories = pq.read_table(self.path_to_data_parquet[:-1]+'_patients/split_group=' + self.split_group + '/').to_pandas()

        total_positive = self.patients_with_valid_trajectories['y'].sum()
        print("Total number of patients  in '{}' dataset is: {}.".format(self.split_group, len(self.patients_with_valid_trajectories)))
        print("Number of positive patients  in '{}' dataset is: {}.".format(self.split_group, total_positive))
        self.class_count()

    def process_events(self, events):
        """
            Process the diagnosis events depending on the filters. If only known risk factors are used,
            then ICD codes that are not in the subset are replaced with PAD token.
        """
        if self.args.use_known_risk_factors_only:
            for e in events:
                if e['codes'] not in self.SETTINGS.KNOWN_RISK_FACTORS and e['codes'] not in self.SETTINGS.PANC_CANCER_CODE:
                    e['codes'] = PAD_TOKEN
        return events
    
    def get_trajectory(self, patient_index):
        """
            Given a patient, multiple trajectories can be extracted by sampling partial histories.
        """
        #we currently do not need the actual patient_id and instead work with the patient_index
        #patient_id is the identifier in MarketScan, whereas patient_index is the index in the patients_with_valid_trajectories table
        patient_id= self.patients_with_valid_trajectories.iloc[patient_index]['patient_id']

        patient_trajectories=self.events.iloc[self.patients_with_valid_trajectories.iloc[patient_index]['first_row']:self.patients_with_valid_trajectories.iloc[patient_index]['last_row']+1].copy()
        patient_trajectories.reset_index(inplace=True)

        #find the indices where the patient has a valid trajectory
        valid_indices = patient_trajectories[patient_trajectories['is_valid_traj']==True].index.tolist()

        if self.split_group in ['dev', 'test', 'attribute']:
            if not self.args.no_random_sample_eval_trajectories:
                selected_idx = [random.choice(valid_indices) for _ in range(self.args.max_eval_indices)]
            else:
                selected_idx = valid_indices[-self.args.max_eval_indices:]

        else:
            selected_idx = [random.choice(valid_indices)]

        samples = []



        for idx in selected_idx:
            events_to_date = patient_trajectories.iloc[:idx + 1]
            last_event = events_to_date.iloc[-1]

            #TODO: if we run into speed issues, we could try to find an elegant solution to move this out of the loop
            future_cancer_tensor = np.zeros((self.num_tasks, 1), dtype=bool)
            for task_idx, key in enumerate(self.CANCER_CODE_dict.keys()):
                future_cancer_tensor[task_idx] = last_event[f'future_{key}_patient']

            deltas_admitdate = np.abs(last_event['admit_date']-events_to_date['admit_date'])
            _, time_seq = self.get_time_seq(deltas_admitdate.values)
            age, age_seq = self.get_time_seq(events_to_date['deltas_age'].values)

            codes = events_to_date['code'].tolist()

            y, y_seq, y_mask, time_at_event, days_to_censor = self.get_label(events_to_date, until_idx=idx)

            samples.append({
                'codes': codes,
                'y': y,
                'y_seq': y_seq,
                'y_mask': y_mask,
                'time_at_event': time_at_event,
                'future_panc_cancer': future_cancer_tensor,
                'patient_id': patient_id, #used to be patient_index
                'days_to_censor': days_to_censor,
                'time_seq': time_seq,
                'age_seq': age_seq,
                'age': age,
                'admit_date': last_event['admit_date']#.isoformat())
            })
        return samples

    def get_time_seq(self, deltas):
        """
            Calculates the positional embeddings depending on the time diff from the events and the reference date.
        """
        multipliers = 2*np.pi / (np.linspace(
            start=MIN_TIME_EMBED_PERIOD_IN_DAYS, stop=MAX_TIME_EMBED_PERIOD_IN_DAYS, num=self.args.time_embed_dim
        ))

        positional_embeddings = np.cos(deltas.reshape(-1, 1) * multipliers.reshape(1, -1))
        return deltas.max(), positional_embeddings


    def class_count(self):
        """
        Calculates the weights used by WeightedRandomSampler for balancing the batches.
        """
        #Todo: at a later stage, we should weigh per cancer - currently we do not balance the cancers between them
        ys = self.patients_with_valid_trajectories['y']
        label_counts = Counter(ys)
        weight_per_label = 1. / len(label_counts)
        label_weights = {
            label: weight_per_label / count for label, count in label_counts.items()
        }
        self.weights = [label_weights[d] for d in ys]

    def get_label(self, events_to_date, until_idx):
        """
        Compute labels for a partial disease trajectory.

        Args:
            events_to_date (DataFrame): The events DataFrame which includes all the processed diagnosis events.
            until_idx (int): Specify the end point for the partial trajectory.

        Returns:
            y (bool): True if the trajectory includes pancreatic cancer diagnosis within the time horizon,
                      False otherwise.
            y_seq (numpy.array): Used as golds in cumulative_probability_layer. An array of zeros with ones from
                                 'time_at_event' to the end, indicating the occurrence of pancreatic cancer diagnosis.
            y_mask (numpy.array): An array indicating how many years are left in the disease window. Contains ones
                                  from the start to 'time_at_event' and zeros for the remaining duration.
                                  (without linear interpolation, y_mask looks like the complement of y_seq)
            time_at_event (int): The position in the time vector (default: [3, 6, 12, 36, 60]) which specifies the
                                 outcome_date.
            days_to_censor (int): Number of days between the outcome_date and the admit_date of the event.

        Examples:
            Ex1:  A partial disease trajectory that includes pancreatic cancer diagnosis between 6-12 months
                  after time of assessment.
                time_at_event: 2
                y_seq: [0, 0, 1, 1, 1]
                y_mask: [1, 1, 1, 0, 0]

            Ex2:  A partial disease trajectory from a patient who never gets pancreatic cancer diagnosis
                  but died between 36-60 months after time of assessment.
                time_at_event: 1
                y_seq: [0, 0, 0, 0, 0]
                y_mask: [1, 1, 1, 1, 0]
        """

        last_event = events_to_date.iloc[until_idx]
        days_to_censor = last_event['outcome_day'] - last_event['admit_date']

        # Initialize multi-task arrays
        y_array = np.zeros(self.num_tasks, dtype=bool)
        y_seq_array = np.zeros((self.num_tasks, self.num_time_steps))
        y_mask_array = np.zeros((self.num_tasks, self.num_time_steps))

        if last_event['is_pos_in_time_horizon']:
            time_at_event = min([i for i, mo in enumerate(self.args.month_endpoints) if days_to_censor < (mo * 30)])
        else:
            time_at_event = self.num_time_steps - 1

        for task_idx, key in enumerate(self.CANCER_CODE_dict.keys()):
            y_array[task_idx] = last_event['is_pos_in_time_horizon'] and last_event[f'future_{key}_patient']

            if y_array[task_idx]:
                y_seq_array[task_idx, time_at_event:] = 1
            y_mask_array[task_idx, :time_at_event + 1] = 1

        return y_array, y_seq_array.astype('float64'), y_mask_array.astype('float64'), time_at_event, days_to_censor

    def __len__(self):
        return len(self.patients_with_valid_trajectories)

    def __getitem__(self, patient_index):

        samples = self.get_trajectory(patient_index)
        items = []
        for sample in samples:
            code_str = " ".join(sample['codes'])
            x = [self.get_index_for_code(code, self.args.code_to_index_map) for code in sample['codes']]
            time_seq = sample['time_seq'].tolist()
            age_seq = sample['age_seq'].tolist()
            item = {
                'x': pad_arr(x, self.args.pad_size, 0),
                'time_seq': pad_arr(time_seq, self.args.pad_size, np.zeros(self.args.time_embed_dim)),
                'age_seq': pad_arr(age_seq, self.args.pad_size, np.zeros(self.args.time_embed_dim)),
                'code_str': code_str
            }
            for key in ['y', 'y_seq', 'y_mask', 'time_at_event', 'admit_date', 'age', 'future_panc_cancer',
                        'days_to_censor', 'patient_id']:
                item[key] = sample[key]
            items.append(item)
        return items

    def get_index_for_code(self, code, code_to_index_map):
        code = get_code(self.args, code)
        pad_index = len(code_to_index_map)
        if code == PAD_TOKEN:
            return pad_index
        if code in code_to_index_map:
            return code_to_index_map[code]
        else:
            return code_to_index_map[UNK_TOKEN]


def pad_arr(arr, max_len, pad_value):
    return np.array([pad_value] * (max_len - len(arr)) + arr[-max_len:])
