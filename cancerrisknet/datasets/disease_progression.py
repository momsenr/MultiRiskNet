from cancerrisknet.datasets.factory import RegisterDataset, UNK_TOKEN, PAD_TOKEN
from cancerrisknet.datasets.filter import get_avai_trajectory_indices
from torch.utils import data
from cancerrisknet.utils.date import parse_date
from cancerrisknet.utils.parsing import get_code, md5, load_data_settings
import tqdm
from collections import Counter
import numpy as np
import random
import pandas as pd
from datetime import datetime

MAX_TIME_EMBED_PERIOD_IN_DAYS = 120 * 365
MIN_TIME_EMBED_PERIOD_IN_DAYS = 10
SUMMARY_MSG = "Constructed disease progression {} dataset with {} records from {} patients, " \
              "and the following class balance:\n  {}"


@RegisterDataset("disease_progression")
class DiseaseProgressionDataset(data.Dataset):
    def __init__(self, args, split_group, data_hdf5_file, preprocess_data=False):
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
        self.data_hdf5_file= data_hdf5_file
        self.SETTINGS = load_data_settings(args)['SETTINGS']

        self.patients = pd.read_hdf(self.data_hdf5_file, key='patients_'+self.split_group)

        if(preprocess_data==True):
            print("Preprocessing {} data...".format(self.split_group))
            self.process_patient_data(save_path='processed_trajectories_'+self.split_group)
        else:
            print("Loading {} data from hard disk...".format(self.split_group))
            self.events= pd.read_hdf(self.data_hdf5_file, key='processed_trajectories_'+self.split_group)

        patients_with_trajectories = self.events.groupby('patient_id').agg({'is_valid_traj': 'sum', 'y': 'max'})
        self.patients_with_valid_trajectories = patients_with_trajectories[
            patients_with_trajectories['is_valid_traj'] > 5]
        total_positive = self.patients_with_valid_trajectories['y'].sum()
        print("Number of positive patients  in '{}' dataset is: {}.".format(self.split_group, total_positive))
        self.class_count()

        self.patients_with_valid_trajectories.reset_index(inplace=True)

    def process_patient_data(self,save_path=None):
        """
            Process patient data and extract valid trajectories.
        """

        #load all events belonging to our split group into memory
        self.events = pd.read_hdf(self.data_hdf5_file, key='diagnosis_'+self.split_group)

        # the next line only is relevant if we base the analysis on known risk factors only
        # events = self.process_events(events_raw)

        # Check if the code is a PANC_CANCER_CODE and mark it as True, otherwise mark it as False
        self.events['is_panc_cancer_code'] = self.events['code'].apply(lambda x: True if (x in self.SETTINGS.PANC_CANCER_CODE) else False)

        # Get a list of indices where the 'is_panc_cancer_code' is True
        cancer_patients = list(self.events.loc[self.events.is_panc_cancer_code == True].index.unique())

        # Check if the index is present in the 'cancer_patients' list and mark it as True, otherwise mark it as False
        self.events['future_panc_cancer_patient'] = np.where(self.events.index.isin(cancer_patients), True, False)

        # Calculate the 'outcome_day' based on conditions using column values
        # If 'is_panc_cancer_code' is False, set 'outcome_day' as the value of 'observation_period_end_day'
        # If 'is_panc_cancer_code' is True, set 'outcome_day' as the value of 'admit_date'
        self.events['outcome_day'] = (1 - self.events['is_panc_cancer_code']) * self.events['observation_period_end_day'] \
                                     + self.events['is_panc_cancer_code'] * self.events["admit_date"]

        # Group the DataFrame by 'patient_id' and find the minimum 'outcome_day' for each patient
        self.events['outcome_day'] = self.events.groupby('patient_id')['outcome_day'].min()

        # Drop the 'observation_period_end_day' column from the DataFrame
        self.events.drop("observation_period_end_day", axis=1, inplace=True)

        """
        The next block checks which trajectories are valid. A trajectory is valid if:
        If the patient is a cancer patient:
         (1) The trajectory must end before the pancreatic cancer event.
         (2) The cancer event must occurr within the certain time after the time of assessment.

        Or if the patient is not a cancer patient:
         (3) The trajectory must end at least args.min_followup_year_if_neg before the end of the dataset
             to exclude those cancer patients died of other reasons with the cancer undetected.
        
        Furthermore, the trajectorie must contain enough events (which we check later)
        """
        self.events['is_pos_pre_cancer'] = self.events["admit_date"] < self.events['outcome_day']
        self.events['is_pos_in_time_horizon'] = (self.events["outcome_day"] - self.events['admit_date'] < max(self.args.month_endpoints)  * 30)
        self.events['is_valid_pos'] = self.events.eval("future_panc_cancer_patient and is_pos_pre_cancer and is_pos_in_time_horizon")
        self.events['enough_min_followup'] = ((self.events["outcome_day"] - self.events['admit_date']) // 365) >= self.args.min_followup_year_if_neg
        self.events['is_valid_neg'] = self.events.eval("not future_panc_cancer_patient and enough_min_followup")
        self.events['is_excluded_traj'] = (self.events['outcome_day'] - self.events['admit_date']) <= 30 * self.args.exclusion_interval
        self.events['is_valid_traj'] = self.events.eval("(not is_excluded_traj) and (is_valid_neg or is_valid_pos)")

        # y indicates whether any of the trajectories include a cancer diagnosis.
        self.events['y'] = self.events.groupby('patient_id')['is_valid_pos'].max()

        if(save_path is not None):
            self.events.to_hdf(self.data_hdf5_file, key=save_path, mode='a')

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
        patient_metadata= self.patients_with_valid_trajectories.iloc[patient_index]
        patient_id= patient_metadata['patient_id']
        patient = self.patients[self.patients.patient_id == patient_id]
        patient_trajectories=self.events[self.events.index == patient_id]
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
            events_to_date = patient_trajectories[:idx + 1]
            last_event = events_to_date.iloc[-1]

            codes = events_to_date['code'].tolist()
            _, time_seq = self.get_time_seq(events_to_date, events_to_date.iloc[-1]['admit_date'])
            age, age_seq = self.get_time_seq(events_to_date, (2007-patient['year_of_birth']*365))
            y, y_seq, y_mask, time_at_event, days_to_censor = self.get_label(events_to_date, until_idx=idx)
            samples.append({
                'codes': codes,
                'y': y,
                'y_seq': y_seq,
                'y_mask': y_mask,
                'time_at_event': time_at_event,
                'future_panc_cancer': last_event['future_panc_cancer'],
                'patient_id': patient_index,
                'days_to_censor': days_to_censor,
                'time_seq': time_seq,
                'age_seq': age_seq,
                'age': age,
                'admit_date': last_event['admit_date']#.isoformat())
            })
        return samples

    def get_time_seq(self, events, reference_date):
        """
            Calculates the positional embeddings depending on the time diff from the events and the reference date.
        """
        events['deltas']=events['admit_date'].apply(lambda x: abs(reference_date - x))
        multipliers = 2*np.pi / (np.linspace(
            start=MIN_TIME_EMBED_PERIOD_IN_DAYS, stop=MAX_TIME_EMBED_PERIOD_IN_DAYS, num=self.args.time_embed_dim
        ))

        #deltas = np.array(events['deltas'])
        #deltas, multipliers = deltas.reshape(len(deltas), 1), multipliers.reshape(1, len(multipliers))
        #positional_embeddings = np.cos(deltas*multipliers)
        positional_embeddings = np.cos(events['deltas'].values.reshape(-1, 1) * multipliers.reshape(1, -1))
        return events['deltas'].max(), positional_embeddings

    def class_count(self):
        """
        Calculates the weights used by WeightedRandomSampler for balancing the batches.
        """
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

        event = events_to_date[until_idx]
        days_to_censor = event['outcome_date'] - event['admit_date']
        num_time_steps= len(self.args.month_endpoints)
        y = event['is_pos_in_time_horizon'] and event['future_panc_cancer']
        y_seq = np.zeros(num_time_steps)
        if event['is_pos_in_time_horizon']:
            time_at_event = min([i for i, mo in enumerate(self.args.month_endpoints) if days_to_censor < (mo*30)])
        else:
            time_at_event = num_time_steps - 1

        if y:
            y_seq[time_at_event:] = 1
        y_mask = np.array([1] * (time_at_event+1) + [0] * (num_time_steps - (time_at_event+1)))

        assert time_at_event >= 0 and len(y_seq) == len(y_mask)
        return y, y_seq.astype('float64'), y_mask.astype('float64'), time_at_event, days_to_censor

    def __len__(self):
        return len(self.patients_with_valid_trajectories)

    def __getitem__(self, patient_index):

        #patient = self.patients_with_valid_trajectories[index]
        samples = self.get_trajectory(patient_index)
        items = []
        for sample in samples:
            code_str = " ".join(sample['code'])
            x = [self.get_index_for_code(code, self.args.code_to_index_map) for code in sample['code']]
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
