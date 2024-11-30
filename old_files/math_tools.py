import pandas as pd
import numpy as np
from numba import jit


def create_atr(data, n=8):
    high_low = data['High'] - data['Low']
    high_prev_close = np.abs(data['High'] - data['Close'].shift(1))
    low_prev_close = np.abs(data['Low'] - data['Close'].shift(1))
    true_range = np.maximum(high_low, high_prev_close)
    true_range = np.maximum(true_range, low_prev_close)

    # Calculate Average True Range (ATR)
    atr = np.zeros_like(data['Close'])
    atr[n - 1] = np.mean(true_range[:n])  # Initial ATR calculation

    for i in range(n, len(data['Close'])):
        atr[i] = ((atr[i - 1] * (n - 1)) + true_range[i]) / n

    return atr


def set_various_data(df):
    df['ATR'] = create_atr(df, 8)/df['Close']*100
    df['EMA'] = calculate_ema_numba(df, 'Close', 8)
    df['EMA_Close'] = (df['Close'] - df['EMA']) / df['Close']*100
    df['EMA'] = standardize_ema(df['EMA'], 8)

    return df


def rsi(series, period):
    delta = series.pct_change()
    delta = delta.dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta <= 0]
    u[u.index[period-1]] = np.mean(u[:period]) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean(d[:period]) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = (pd.DataFrame.ewm(u, com=period-1, adjust=False).mean() /
          pd.DataFrame.ewm(d, com=period-1, adjust=False).mean())

    return 100 - 100 / (1 + rs)


def create_rsi(df, securities):
    for sec in securities:
        rsi_k_name = f'{sec}_RSI_k'
        rsi_d_name = f'{sec}_RSI_d'
        df[rsi_k_name] = np.hstack([[0]*14, rsi(df[f'{sec}_Close'], 14)])
        df[rsi_d_name] = df[rsi_k_name].rolling(window=9).mean()

    return df


def add_high_low_diff(df, other_sec, sec_name):
    print('...adding high-low diff and ratio')
    securities = other_sec + [sec_name]
    for sec in securities:
        df[f'{sec}_HL_diff'] = (
                df[f'{sec}_High'] - df[f'{sec}_Low']) / ((df[f'{sec}_High'] + df[f'{sec}_Low'])/2)*100
        df[f'{sec}_OC_diff'] = (
                df[f'{sec}_Open'] - df[f'{sec}_Close']) / ((df[f'{sec}_Open'] + df[f'{sec}_Close'])/2)*100

        df[f'{sec}_HL_Ratio'] = df[f'{sec}_HL_diff'] / df[f'{sec}_HL_diff'].shift(1)
        df[f'{sec}_OC_Ratio'] = df[f'{sec}_OC_diff'] / df[f'{sec}_OC_diff'].shift(1)

    drop_cols = [f'{sec}_{col}' for sec in other_sec for col in ['High', 'Low']]
    df.drop(columns=drop_cols, inplace=True)

    return df


def scale_open_close(df):
    print('...scaling open, close')
    for col in df.columns.to_list():
        if any(word in col for word in ['Open', 'High', 'Close', 'Low']):
            df[col] = df[col]/df[col].shift(1)

    return df


def smooth_vol_oi(df, securities):
    print('...smoothing vol, oi')
    for sec in securities:
        volavg = df[f'{sec}_Vol'] / (df[f'{sec}_Vol'].rolling(window=23, min_periods=1).mean())
        df[f'{sec}_Vol_Avg'] = volavg

        volchng = df[f'{sec}_Vol'] / df[f'{sec}_Vol'].shift(1)
        df[f'{sec}_Vol_Chng'] = volchng.fillna(0)

        oi_avg = df[f'{sec}_OpenInt'] / (df[f'{sec}_OpenInt'].rolling(window=23, min_periods=1).mean())
        df[f'{sec}_OpenInt'] = oi_avg.fillna(0)

        oi_avg = df[f'{sec}_OpenInt']/df[f'{sec}_OpenInt'].shift(1)
        df[f'{sec}_Oi_Chng'] = oi_avg.fillna(0)

    return df


def calculate_max_drawdown(pnl_series):
    draw_list = [0]
    arr = pnl_series.values
    for i in range(1, len(arr)):
        prev_max = np.max(arr[:i])
        draw_list.append(min(0, arr[i] - prev_max, arr[i-1]))

    if len(draw_list) > 1:
        draw_list.pop(0)
        draw_list.append(draw_list[-1])

    return draw_list


def calculate_algo_lstm_ratio(algo_series, lstm_series, max_lever):
    draw_ratio_list = [1]
    algo_arr = algo_series.values
    lstm_arr = lstm_series.values
    for i in range(1, (len(algo_arr))):
        start_ind = max(0, i-50)
        if np.min(lstm_arr[start_ind:i]) == 0:
            draw_ratio_list.append(1)
        else:
            algo_1 = algo_arr[start_ind:i]
            algo_1 = algo_1[algo_1 != 0]

            lstm_1 = lstm_arr[start_ind:i]
            lstm_1 = lstm_1[lstm_1 != 0]

            if (len(lstm_1) == 0) or (len(algo_1) == 0):
                draw_ratio_list.append(draw_ratio_list[-1])
            else:
                max_draw = np.median(algo_1)/np.median(lstm_1)
                draw_ratio_list.append(max(1, min(max_draw, max_lever)))

    return draw_ratio_list


def calculate_ema_numba(df, price_colname, window_size, smoothing_factor=2):
    result = calculate_ema_inner(
        price_array=df[price_colname].to_numpy(),
        window_size=window_size,
        smoothing_factor=smoothing_factor
    )

    # return pd.Series(result, index=df.index, name="result", dtype=float)
    # result = np.array(standardize_ema(result))

    return result


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


def standardize_ema(arr, lag=12):
    # Create an output array of the same length, initialized to 1
    arr = np.array(arr)
    standardized_arr = np.ones_like(arr, dtype=float)

    # For elements that have at least `lag` elements before them
    standardized_arr[lag:] = arr[lag:] / arr[:-lag]

    return standardized_arr


def summary_predicted(df, wl=False):
    df['PnL'] = df['PnL']*df['Close']/100
    df['Algo_PnL_Total'] = df['PnL'].cumsum()
    df['Algo_MaxDraw'] = calculate_max_drawdown(df['Algo_PnL_Total'])
    if wl:
        df['Pred_PnL'] = np.where(df['Pred'] == 'Loss', 0, df['PnL'])
    else:
        df['Pred_PnL'] = np.where(df['Pred'] < 0, 0, df['PnL'])
    df['Pred_PnL_Total'] = df['Pred_PnL'].cumsum()
    df['Pred_MaxDraw'] = calculate_max_drawdown(df['Pred_PnL_Total'])

    df.fillna(0, inplace=True)

    return df












