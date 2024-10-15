import pandas as pd
import numpy as np


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
    for c in ['ATR']:
        df[c] = df[c]/df['Close']*100

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
                df[f'{sec}_High'] - df[f'{sec}_Low'])/((df[f'{sec}_High'] + df[f'{sec}_Low'])/2)*1000
        df[f'{sec}_OC_diff'] = (
                df[f'{sec}_Open'] - df[f'{sec}_Close'])/((df[f'{sec}_Open'] + df[f'{sec}_Close'])/2)*1000

        df[f'{sec}_HL_Ratio'] = df[f'{sec}_HL_diff']/df[f'{sec}_HL_diff'].shift(1)
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
        volsmooth = df[f'{sec}_VolAvg']/df[f'{sec}_VolAvg'].shift(1)
        df[f'{sec}_VolAvg'] = volsmooth.fillna(0)

        df[f'{sec}_Vol'] = df[f'{sec}_Vol']/df[f'{sec}_Vol'].rolling(window=23, min_periods=1).mean()

        oi_avg = df[f'{sec}_OpenInt'].rolling(window=23, min_periods=1).mean()
        oi_avg = oi_avg/oi_avg.shift(1)
        df[f'{sec}_OpenInt'] = oi_avg.fillna(0)

    return df


def calculate_max_drawdown(pnl_series):
    pnl_series = pnl_series.values
    max_drawdown = []
    peak = pnl_series[0]  # Initial peak

    for pnl in pnl_series:
        if pnl > peak:
            peak = pnl  # Update peak if current pnl is higher
        drawdown = (peak - pnl) / peak  # Calculate drawdown
        max_drawdown.append(drawdown)

    return max_drawdown


def summary_predicted(df, wl=False):
    df['PnL'] = df['PnL']*df['Close']
    df['Algo_PnL_Total'] = df['PnL'].cumsum()
    df['Algo_MaxDraw'] = calculate_max_drawdown(df['Algo_PnL_Total'])
    if wl:
        df['Pred_PnL'] = np.where(df['Pred'] == 'Loss', 0, df['PnL'])
    else:
        df['Pred_PnL'] = np.where(df['Pred'] < 0, 0, df['PnL'])
    df['Pred_PnL_Total'] = df['Pred_PnL'].cumsum()
    df['Pred_MaxDraw'] = calculate_max_drawdown(df['Pred_PnL_Total'])
    df['Pred_PnL_Adj'] = df['Pred_PnL']*(max(df['Algo_MaxDraw'])/max(df['Pred_MaxDraw']))
    df.fillna(0, inplace=True)

    return df












