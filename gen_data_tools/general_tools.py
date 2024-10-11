import numpy as np
import pandas as pd
from datetime import timedelta
from dataclasses import dataclass


"""---------------------------------------------Aggregate Work-------------------------------------------------------"""
def adjust_dates(dates):
    try:
        date = pd.to_datetime(dates, format='%Y-%m-%d')
    except ValueError:
        try:
            date = pd.to_datetime(dates, format='%m/%d/%Y')
        except ValueError:
            date = pd.to_datetime(dates, format='%Y/%m/%d')
        
    return date


def adjust_datetime(datetimes):
    datetimes = pd.to_datetime(datetimes, format='%Y-%m-%d %H:%M:%S')
    return datetimes


def convert_date_time(data):
    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data['Date'] = pd.to_datetime(data['Date']).dt.date

    return data


def create_datetime(df):
    datetimes = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M')

    return datetimes


def fix_tf_arrays(arr):
    arr = arr.astype(int)

    return arr


def subset_time(df, dbl_setup, subtract_time=0):
    """Subsets for start_time and end_time. Option to subtract an hour from start_time in order to allow some
    space for setups. This means that you don't have to run on the full dataset to get accurate info for the
    targetted time"""

    eod_mask = (df['DateTime'].dt.time >=
                (dbl_setup.start_time - timedelta(hours=subtract_time)).time()) & \
               (df['DateTime'].dt.time <= dbl_setup.eod_time.time())

    return df[eod_mask]


def filter_trades(df):
    df = df.loc[(df['bullTrade'] == 1) | (df['bearTrade'] == 1)]
    df = df[['DateTime', 'side', 'entryInd', 'entryPrice', 'exitInd', 'exitPrice']]
    df.reset_index(drop=True, inplace=True)

    return df


def get_side(df):
    side = np.where(df['bullTrade'] == 1, 'Bull', np.where(df['bearTrade'] == 1, 'Bear', ''))

    return side


def sort_data_cols(df):
    sorted_cols = sorted(df.columns.to_list()[1:], key=lambda x: (not x.startswith('NQ'), x))
    df = df[[df.columns.to_list()[0]] + sorted_cols]
    return df


def fill_na_inf(df):
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df


def pad_to_length(arr, target_length):
    current_length, num_columns = arr.shape
    padded_columns = []
    for i in range(num_columns):
        column = arr[:, i]
        padding_needed = target_length - current_length

        if padding_needed > 0:
            padded_column = np.pad(column, (padding_needed, 0), 'constant')
        else:
            padded_column = column

        padded_columns.append(padded_column)

    padded_array = np.column_stack(padded_columns)

    return padded_array

