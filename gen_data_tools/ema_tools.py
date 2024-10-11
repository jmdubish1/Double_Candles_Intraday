import numpy as np
import pandas as pd
from numba import jit
import os


def check_create_emas(data_class, ema_range, daily=False):
    setup_class = data_class.setup_params
    if daily:
        file_output = (f'{setup_class.strat_loc}\\{setup_class.security}\\'
                       f'{setup_class.timeframe}\\{setup_class.security}_daily_EMAs.csv')
    else:
        file_output = (f'{setup_class.strat_loc}\\{setup_class.security}\\'
                       f'{setup_class.timeframe}\\{setup_class.security}_{setup_class.timeframe}_EMAs.csv')

    file_exists = os.path.exists(file_output)
    if file_exists:
        if daily:
            ema_df = pd.read_csv(file_output)
            return ema_df
        else:
            return get_missing_cols(file_output, ema_range)
    else:
        if daily:
            ema_df = create_ema_df(data_class.daily_df, ema_range, daily)
            ema_df.to_csv(file_output, index=False)
            return ema_df
        else:
            ema_df = create_ema_df(data_class.clean_df, ema_range, daily)
            ema_df.to_csv(file_output, index=False)
            return ema_df


def get_missing_cols(file_output, ema_range):
    ema_df = pd.read_csv(file_output)
    col_list = ema_df.columns
    ema_list = list(ema_range)
    missing_columns = [col for col in ema_list if f'EMA_{col}' not in col_list]

    if len(missing_columns) > 0:
        for missing_ema in missing_columns:
            ema_df.loc[:, f'EMA_{missing_ema}'] = calculate_ema_numba(
                df=ema_df,
                price_colname='Open',
                window_size=missing_ema
            )
        ema_df.to_csv(file_output, index=False)

    return ema_df


def create_ema_df(df, ema_len_list, daily=False):
    for ema_len in ema_len_list:
        df.loc[:, f'EMA_{ema_len}'] = calculate_ema_numba(
            df=df,
            price_colname='Open',
            window_size=ema_len
        )

    if daily:
        df = df[['Date', 'Open', 'EMA_8']]
    else:
        df = df[['DateTime', 'Open'] + [f'EMA_{ema_len}' for ema_len in ema_len_list]]

    return df


def calculate_ema_numba(df, price_colname, window_size, smoothing_factor=2):
    result = calculate_ema_inner(
        price_array=df[price_colname].to_numpy(),
        window_size=window_size,
        smoothing_factor=smoothing_factor
    )

    return pd.Series(result, index=df.index, name="result", dtype=float)


@jit(nopython=True)
def calculate_ema_inner(price_array, window_size, smoothing_factor):
    result = np.empty(len(price_array), dtype="float64")
    sma_list = list()
    for i in range(len(result)):

        if i < window_size - 1:
            # assign NaN to row, append price to simple moving average list
            result[i] = np.nan
            sma_list.append(price_array[i])
        elif i == window_size - 1:
            # calculate simple moving average
            sma_list.append(price_array[i])
            result[i] = sum(sma_list) / len(sma_list)
        else:
            # compute exponential moving averages according to formula
            result[i] = ((price_array[i] * (smoothing_factor / (window_size + 1))) +
                         (result[i - 1] * (1 - (smoothing_factor / (window_size + 1)))))

    return result


def create_ema_df(df, ema_len_list, daily=False):
    for ema_len in ema_len_list:
        print(ema_len)
        df.loc[:, f'EMA_{ema_len}'] = calculate_ema_numba(
            df=df,
            price_colname='Open',
            window_size=ema_len
        )

    if daily:
        df = df[['Date', 'Open'] + [f'EMA_{ema_len}' for ema_len in ema_len_list]]
    else:
        df = df[['DateTime', 'Open'] + [f'EMA_{ema_len}' for ema_len in ema_len_list]]

    return df
