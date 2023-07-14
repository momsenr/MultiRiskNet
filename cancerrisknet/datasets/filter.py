from cancerrisknet.utils.date import parse_date

def get_avai_trajectory_indices(patient, events, args):
    """
        This function takes a patients and its events as pandas dataframe and returns all rows that are valid
        trajectories depending on the filters applied.

        Filters implemented in this version:
            - Exclusion interval: Removes events too close to PC. If exclusion interval is 0 then do not remove. 

        Returns:
            valid_trajectories (pandas df): Each row is corresponding to one partial trajectory that passed the filter.
            y (bool): If valid_indices is not empty, then y indicates whether any of the trajectories
                      include a cancer diagnosis.

    """
    y = False

    #add column is_valid_idx to df
    events['is_valid_idx'] = False

    for idx, row in events.iterrows():
        if patient['future_panc_cancer'] and \
                (patient['outcome_date'] - row['admit_date']).days <= 30 * args.exclusion_interval:
            continue

        if is_valid_trajectory(events.iloc[:idx+1], patient['outcome_date'], patient['future_panc_cancer'], args):
            row['is_valid_idx'] = True
            days_to_censor = (patient['outcome_date'] - row['admit_date']).days
            y = (days_to_censor < (max(args.month_endpoints) * 30) and patient['future_panc_cancer']) or y

    return events, y


def is_valid_trajectory(events_to_date, outcome_date, future_panc_cancer, args):
    """
    This function checks whether a single trajectory is valid. A trajectory is valid if:
     (1) It contains enough events.

    And if the patient is a cancer patient,
     (2) The trajectory must end before the pancreatic cancer event.
     (3) The cancer event must occurr within the certain time after the time of assessment.

    Or if the patient is not a cancer patient
     (4) The trajectory must end at least args.min_followup_year_if_neg before the end of the dataset
         to exclude those cancer patients died of other reasons with the cancer undetected.

    """

    # Filter (1)
    enough_events_counted = len(events_to_date) >= args.min_events_length
    if not enough_events_counted:
        return False

    # Filter (2-3)
    #todo: vectorize date conversion
    is_pos_pre_cancer = events_to_date.iloc[-1]['admit_date']< outcome_date
    is_pos_in_time_horizon = (outcome_date - events_to_date.iloc[-1]['admit_date']).days < max(args.month_endpoints) * 30
    is_valid_pos = future_panc_cancer and is_pos_pre_cancer and is_pos_in_time_horizon

    # Filter (4)
    is_valid_neg = not future_panc_cancer and \
        (outcome_date - events_to_date.iloc[-1]['admit_date']).days // 365 > args.min_followup_year_if_neg

    return is_valid_neg or is_valid_pos
