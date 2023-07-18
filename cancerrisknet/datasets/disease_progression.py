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

        self.patients = pd.read_hdf(self.data_hdf5_file, key='patients')

        #this conversion is necessary, see https://stackoverflow.com/questions/39278042/storing-pure-python-datetime-datetime-in-pandas-dataframe
        #by this we force datetime time (instead of pandas timestamp)
        #self.arr_date_patients= self.patients['observation_period_end_date'].dt.to_pydatetime()
        #self.patients['observation_period_end_date_tf']= pd.Series(self.arr_date_patients, dtype=object)
        
        if(preprocess_data==True):
            self.process_patient_data(save_path='patients_with_valid_trajectories'+self.split_group)
        else:
            self.patients_with_valid_trajectories= pd.read_hdf(self.data_hdf5_file, key='patients_with_valid_trajectories')

    def process_patient_data(self,save_path=None):
        """
            Process patient data and extract valid trajectories.
        """
        #load all events into memory
        self.events = pd.read_hdf(self.data_hdf5_file, 'diagnosis')

        # the next line only is relevant if we base the analysis on known risk factors only
        # events = self.process_events(events_raw)

        #create new pandas dataframe in which we store the metadata
        #self.patients_with_valid_trajectories = pd.DataFrame(columns=['patient_id', 'dob', 'future_panc_cancer', 'outcome_date', 'obs_time_end', 'y'])

        # Check if the code is '157' or 'C25' and mark it as True, otherwise mark it as False
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

        # Determine valid trajectories
        self.events['is_pos_pre_cancer'] = self.events["admit_date"] < self.events['outcome_day']
        self.events['is_pos_in_time_horizon'] = (self.events["outcome_day"] - self.events['admit_date'] < max(self.args.month_endpoints)  * 30)
        self.events['is_valid_pos'] = self.events.eval("future_panc_cancer_patient and is_pos_pre_cancer and is_pos_in_time_horizon")
        self.events['enough_min_followup'] = ((self.events["outcome_day"] - self.events['admit_date']) // 365) >= self.args.min_followup_year_if_neg
        self.events['is_valid_neg'] = self.events.eval("not future_panc_cancer_patient and enough_min_followup")
        self.events['is_excluded_traj'] = (self.events['outcome_day'] - self.events['admit_date']) <= 30 * self.args.exclusion_interval
        self.events['is_valid_traj'] = self.events.eval("(not is_excluded_traj) and (is_valid_neg or is_valid_pos)")

        self.events['y'] = self.events.groupby('patient_id')['is_valid_pos'].max()

        if(save_path is not None):
            self.events.to_hdf5(self.data_hdf5_file, key=save_path, mode='a')

        patients_with_trajectories = self.events.groupby('patient_id').agg({'is_valid_traj': 'sum', 'y': 'max'})
        self.patients_with_valid_trajectories= patients_with_trajectories[patients_with_trajectories['is_valid_traj'] > 0]
        total_positive = self.patients_with_valid_trajectories['y'].sum()
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

    def get_trajectory(self, patient):
        """
            Given a patient, multiple trajectories can be extracted by sampling partial histories.
        """

        valid_trajectories = pd.read_hdf(self.data_hdf5_file, key='valid_trajectories_df'+self.split_group, where='index=='+patient['patient_id'])

        #find all indices where the patient has a valid trajectory
        valid_indices = valid_trajectories[valid_trajectories['is_valid_idx']==1].index.tolist()

        if self.split_group in ['dev', 'test', 'attribute']:
            if not self.args.no_random_sample_eval_trajectories:
                selected_idx = [random.choice(valid_indices) for _ in range(self.args.max_eval_indices)]
            else:
                selected_idx = valid_indices[-self.args.max_eval_indices:]

        else:
            selected_idx = [random.choice(valid_indices)]

        samples = []
        for idx in selected_idx:
            events_to_date = valid_trajectories[:idx + 1]

            codes = events_to_date['codes'].tolist()
            _, time_seq = self.get_time_seq(events_to_date, events_to_date[-1]['admit_date'])
            age, age_seq = self.get_time_seq(events_to_date, patient['dob'])
            y, y_seq, y_mask, time_at_event, days_to_censor = self.get_label(patient, until_idx=idx)
            samples.append({
                'codes': codes,
                'y': y,
                'y_seq': y_seq,
                'y_mask': y_mask,
                'time_at_event': time_at_event,
                'future_panc_cancer': patient['future_panc_cancer'],
                'patient_id': patient['patient_id'],
                'days_to_censor': days_to_censor,
                'time_seq': time_seq,
                'age_seq': age_seq,
                'age': age,
                'admit_date': events_to_date[-1]['admit_date'].isoformat(),
                'exam': str(events_to_date[-1]['admid'])
            })
        return samples

    def get_time_seq(self, events, reference_date):
        """
            Calculates the positional embeddings depending on the time diff from the events and the reference date.
        """
        deltas = np.array([abs((reference_date - event['admit_date']).days) for event in events])
        multipliers = 2*np.pi / (np.linspace(
            start=MIN_TIME_EMBED_PERIOD_IN_DAYS, stop=MAX_TIME_EMBED_PERIOD_IN_DAYS, num=self.args.time_embed_dim
        ))

        deltas, multipliers = deltas.reshape(len(deltas), 1), multipliers.reshape(1, len(multipliers))
        positional_embeddings = np.cos(deltas*multipliers)
        return max(deltas), positional_embeddings

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

    def get_label(self, patient, until_idx):
        """
        Args:
            patient (dict): The patient dictionary which includes all the processed diagnosis events.
            until_idx (int): Specify the end point for the partial trajectory.

        Returns:
            outcome_date: date of pancreatic cancer diagnosis for cases (cancer patients) or
                          END_OF_TIME_DATE for controls (normal patients)
            time_at_event: the position in time vector (default: [3,6,12,36,60]) which specify the outcome_date
            y_seq: Used as golds in cumulative_probability_layer
                   An all zero array unless ever_develops_panc_cancer then y_seq[time_at_event:]=1
            y_mask: how many years left in the disease window
                    ([1] for 0:time_at_event years and [0] for the rest)
                    (without linear interpolation, y_mask looks like complement of y_seq)

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
        event = patient['events'][until_idx]
        days_to_censor = (patient['outcome_date'] - event['admit_date']).days
        num_time_steps, max_time = len(self.args.month_endpoints), max(self.args.month_endpoints)
        y = days_to_censor < (max_time*30) and patient['future_panc_cancer']
        y_seq = np.zeros(num_time_steps)
        if days_to_censor < (max_time * 30):
            time_at_event = min([i for i, mo in enumerate(self.args.month_endpoints) if days_to_censor < (mo*30)])
        else:
            time_at_event = num_time_steps - 1

        if y:
            y_seq[time_at_event:] = 1
        y_mask = np.array([1] * (time_at_event+1) + [0] * (num_time_steps - (time_at_event+1)))

        assert time_at_event >= 0 and len(y_seq) == len(y_mask)
        return y, y_seq.astype('float64'), y_mask.astype('float64'), time_at_event, days_to_censor

    def get_outcome_date(self, events, end_of_date=None):
        """
        Looks through events to find date of outcome, which is defined as either pancreatic cancer
        occurrence time or the end of trajectory. If multiple cancer events exist, use the first diagnosis date.

        Args:
            events: A pandas df where each row must have a code and admit_date.
            end_of_date: The date for the death for the patient or the end date for
                         the entire dataset (e.g. the patient is still alive).

        Returns:
            ever_develops_panc_cancer (bool): Assess if any given partial trajectory has at least
                                              one diagnosis of pancreatic cancer.
            time (datetime): The Date of pancreatic cancer diagnosis for cases (cancer patients) or
                             END_OF_TIME_DATE for controls (normal patients)

        """
        if end_of_date is None:
            end_of_date = self.SETTING.END_OF_TIME_DATE
        panc_ca_events = events[events['code'].isin(self.SETTINGS.PANC_CANCER_CODE)]

        if not panc_ca_events.empty:
            ever_develops_panc_cancer = True
            time = panc_ca_events['admit_date'].min()
        else:
            ever_develops_panc_cancer = False
            time = end_of_date
        return ever_develops_panc_cancer, time

    def __len__(self):
        return len(self.patients_with_valid_trajectories)

    def __getitem__(self, index):

        patient = self.patients_with_valid_trajectories[index]
        samples = self.get_trajectory(patient)
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
            for key in ['y', 'y_seq', 'y_mask', 'time_at_event', 'admit_date', 'exam', 'age', 'future_panc_cancer',
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
