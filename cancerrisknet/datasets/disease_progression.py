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
import json

MAX_TIME_EMBED_PERIOD_IN_DAYS = 120 * 365
MIN_TIME_EMBED_PERIOD_IN_DAYS = 10

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

        with open(self.args.cancer_code_dict_path, 'r') as file:
            data = file.read()
        self.CANCER_CODE_dict = json.loads(data)
        self.num_tasks = len(self.CANCER_CODE_dict)
        self.num_classees = self.num_tasks + 1

        self.num_time_steps= len(self.args.month_endpoints)

        if(preprocess_data==True):
            print('not implemented in the DataLoader currently...')
            sys.exit(-1)
        else:
            print("Loading {} data from hard disk...".format(self.split_group))
            self.events=pq.read_table(self.path_to_data_parquet+self.split_group+'_processed/').to_pandas()

            #while in an earlier sorting was not necessary, we now have to restore the row order
            #possibly due to pyarrow update
            self.events.sort_values(by=['patient_id', 'admit_date', 'is_valid_traj'],inplace=True)

            #we do not need the patient_id anymore
            self.events.drop(columns=['patient_id'], inplace=True)

            self.patients_with_valid_trajectories = pq.read_table(self.path_to_data_parquet + self.split_group + '_patients/').to_pandas()

        print("Total number of patients  in '{}' dataset is: {}.".format(self.split_group, len(self.patients_with_valid_trajectories)))
        for key in self.CANCER_CODE_dict.keys():
            total_positive = self.patients_with_valid_trajectories[f'y_{key}'].sum()
            print("Number of positive patients for cancer type '{}' in '{}' dataset is: {}.".format(key, self.split_group, total_positive))

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

        patient= self.patients_with_valid_trajectories.iloc[patient_index]
        patient_id= patient['patient_id']
        patient_trajectories=self.events.iloc[self.patients_with_valid_trajectories.iloc[patient_index]['first_row']:self.patients_with_valid_trajectories.iloc[patient_index]['last_row']+1].copy()
        patient_trajectories.reset_index(inplace=True)

        #find the indices where the patient has a valid trajectory
        valid_indices = patient_trajectories[patient_trajectories['is_valid_traj']==True].index.tolist()

        if self.split_group in ['dev', 'test', 'attribute']:
            if not self.args.no_random_sample_eval_trajectories:
                selected_idx = random.sample(valid_indices, min(len(valid_indices), self.args.max_eval_indices))
            else:
                selected_idx = valid_indices[-self.args.max_eval_indices:]

        else:
            selected_idx = [random.choice(valid_indices)]
        samples = []


        future_cancer_array = np.zeros((self.num_tasks, 1), dtype=bool)
        for task_idx, key in enumerate(self.CANCER_CODE_dict.keys()):
            future_cancer_array[task_idx] = patient[f'y_{key}']

        for idx in selected_idx:
            events_to_date = patient_trajectories.iloc[:idx + 1]
            last_event = events_to_date.iloc[-1]

            deltas_admitdate = np.abs(last_event['admit_date']-events_to_date['admit_date'])
            _, time_seq = self.get_time_seq(deltas_admitdate.values)
            age, age_seq = self.get_time_seq(events_to_date['deltas_age'].values)

            codes = events_to_date['code'].tolist()

            y_seq, y_mask, censor_time_index, days_to_censor = self.get_label(events_to_date, until_idx=idx)

            samples.append({
                'codes': codes,
                'y_seq': y_seq,
                'y_mask': y_mask,
                'censor_time_index': censor_time_index,
                'future_cancer_array': future_cancer_array,
                'patient_id': patient_id,
                'days_to_censor': days_to_censor,
                'time_seq': time_seq,
                'age_seq': age_seq,
                'age': age,
                'admit_date': last_event['admit_date']
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
        return deltas.max().astype(int), positional_embeddings


    def class_count(self):
        """
        Calculates the weights used by WeightedRandomSampler for balancing the batches.
        """
        # Initialize a list to store arrays of weights for each task
        task_weights = []
        self.patients_with_valid_trajectories['label'] = 0

        # Update the 'label' column based on the rules
        for idx, key in enumerate(self.CANCER_CODE_dict.keys()):
            y_key_col = f'y_{key}'
    
            # Update 'label' to idx+1 where 'label' is 0 and 'y_{key}' is True
            mask = (self.patients_with_valid_trajectories['label'] == 0) & (self.patients_with_valid_trajectories[y_key_col] == True)
            self.patients_with_valid_trajectories.loc[mask, 'label'] = idx + 1
    
        ys = self.patients_with_valid_trajectories['label']
        self.patients_with_valid_trajectories.drop(columns=['label'], inplace=True)
        label_counts = Counter(ys)
        # Define your desired ratios
        if(self.args.loss_weights=='PC'):
            desired_ratios = {'PC': 2, 'OC': 1, '0': 1}
        else:
            desired_ratios = {'PC': 1, 'OC': 1, '0': 1}

        # Adjusting the label_weights calculation
        label_weights = {}
        for label, count in label_counts.items():
            # Assuming label 1 corresponds to PC, label 2 to OC, and label 0 to class 0
            if label == 1:  # PC
                ratio_factor = desired_ratios['PC']
            elif label == 2:  # OC
                ratio_factor = desired_ratios['OC']
            else:  # Class 0
                ratio_factor = desired_ratios['0']

            label_weights[label] = (1.0 / count) * ratio_factor

        #weight_per_label = 1. / len(label_counts)
        #label_weights = {
        #    label: weight_per_label / count for label, count in label_counts.items()
        #}
        self.weights = [label_weights[d] for d in ys]

    def get_label(self, events_to_date, until_idx):
        """
        Compute labels for a partial disease trajectory.

        Args:
            events_to_date (DataFrame): The events DataFrame which includes all the processed diagnosis events.
            until_idx (int): Specify the end point for the partial trajectory.

        Returns:
            y_seq_array (numpy.array): A 1D integer array indicating the class index of cancer types,
                                including a 'no cancer' class. If a cancer diagnosis occurs within the
                                time horizon, the corresponding class index is set at the time index of
                                the diagnosis and afterwards. Shape is (num_time_steps,).
            y_mask_array (numpy.array): A boolean array indicating the time until the cancer diagnosis.
                                        Contains ones from the start to 'censor_time_index' for the cancer
                                        that occurs and zeros thereafter. Shape is (num_time_steps,).
            censor_time_index (integer): An integers specifying the position in the time vector at which the trajectory
                                        is censored.
            days_to_censor_array (numpy.array): An array of integers representing the number of days between the
                                                outcome_date and the admit_date of each event. Shape is (num_tasks,).

        Examples:
            # Example usage of the function...

        """

        last_event = events_to_date.iloc[until_idx]
        days_to_censor_array = np.zeros(self.num_tasks, dtype=int)
        time_index_at_event_array = np.zeros(self.num_tasks, dtype=int)

        # Initialize arrays
        y_seq_array = np.zeros(self.num_time_steps, dtype=int)
        y_mask_array = np.zeros(self.num_time_steps, dtype=bool)

        y_seq_array[:] = self.num_tasks #set all entries to the no cancer class
        y_mask_array[:] = True

        for task_idx, key in enumerate(self.CANCER_CODE_dict.keys()):
            days_to_censor_array[task_idx] = last_event[f'outcome_day_{key}'] - last_event['admit_date']
            if last_event[f'is_pos_{key}_in_time_horizon']:
                time_index_at_event_array[task_idx] = min([i for i, mo in enumerate(self.args.month_endpoints) if days_to_censor_array[task_idx] < (mo * 30)])
            else:
                time_index_at_event_array[task_idx] = self.num_time_steps - 1

        #todo: this does not yield the correct result if a patient is first positive for one cancer,
        # and then for the other.
        for task_idx, key in enumerate(self.CANCER_CODE_dict.keys()):
            if(last_event[f'is_pos_{key}_in_time_horizon'] and last_event[f'future_{key}_patient']):
                y_seq_array[time_index_at_event_array[task_idx]:] = task_idx
            
        censor_time_index=time_index_at_event_array.min()
        y_mask_array[censor_time_index + 1:] = False

        #todo: also take the minimum of days_to_censor_array?

        return y_seq_array, y_mask_array, censor_time_index, days_to_censor_array   
        
    def __len__(self):
        return len(self.patients_with_valid_trajectories)

    def __getitem__(self, patient_index):

        samples = self.get_trajectory(patient_index)
        items = []
        for sample in samples:
            #code_str = " ".join(sample['codes'])
            code_str = " ".join([str(code) for code in sample['codes']])
            x = [self.get_index_for_code(code, self.args.code_to_index_map) for code in sample['codes']]
            time_seq = sample['time_seq'].tolist()
            age_seq = sample['age_seq'].tolist()
            item = {
                'x': pad_arr(x, self.args.pad_size, 0),
                'time_seq': pad_arr(time_seq, self.args.pad_size, np.zeros(self.args.time_embed_dim)),
                'age_seq': pad_arr(age_seq, self.args.pad_size, np.zeros(self.args.time_embed_dim)),
                'code_str': code_str
            }
            for key in ['y_seq', 'y_mask', 'censor_time_index', 'admit_date', 'age', 'future_cancer_array',
                        'days_to_censor', 'patient_id']:
                item[key] = sample[key]
            items.append(item)
        return items

    def get_index_for_code(self, code, code_to_index_map):
        #2024-01-15: we now moved all the preprocessing to the jupyter notebooks
        #code = get_code(self.args, code)
        pad_index = len(code_to_index_map)
        if code == PAD_TOKEN:
            return pad_index
        if code in code_to_index_map:
            return code_to_index_map[code]
        else:
            return code_to_index_map[UNK_TOKEN]


def pad_arr(arr, max_len, pad_value):
    return np.array([pad_value] * (max_len - len(arr)) + arr[-max_len:])
