import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings(action="ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)


def adj_columns(df):
    for col in df.columns[1:]:
        df[col] = df[col].astype(np.float32)

    return df


def get_weights_FFD(d, thres=1e-5):
    """Use a threshold to define a length of weights"""
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres: break
        w.append(w_); k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff_FFD(arr, d, thres, window=100):
    """Vectorized
    ensure array has been cleared of nan"""
    weights = get_weights_FFD(d, thres)
    result = np.convolve(arr, weights.flatten(), mode='valid')[:len(arr)]
    result = np.pad(result, (len(weights) - 1, 0), constant_values=np.nan)

    result[:window] = np.nan

    return result

def subset_to_first_nonzero(arr):
    first_nonz_ind = np.argmax(arr != 0)
    trimmed_arr = arr[first_nonz_ind:]

    return trimmed_arr


def adf_d_FFD_matrix(arr, test_len=11, log_scale=True, thres=1e-5):
    if log_scale:
        arr = np.log(arr)
    arr = pd.Series(arr).fillna(method='ffill').dropna().values

    adf_data = []
    test_arrs = []
    for d in np.linspace(0, 1, test_len):
        test_arr = frac_diff_FFD(arr, d, thres)
        test_arr[np.isnan(test_arr)] = 0

        if len(set(test_arr)) == 1:
            print(f'No valid values for thres: {thres}')
            continue

        test_arr = subset_to_first_nonzero(test_arr)
        arr_1 = arr[-len(test_arr):]

        corr = np.corrcoef(arr_1, test_arr)[0, 1]
        adf = adfuller(test_arr, maxlag=12, autolag='AIC')
        adf = [i for i in adf]
        adf.append(corr)
        adf.append(d)
        adf.append(len(test_arr))
        test_arrs.append([d, thres, test_arr])

        adf_data.append(adf)

    return adf_data, test_arrs


def organize_adf_data(adf_data):
    adf_df = pd.DataFrame(adf_data)
    adf_df.columns = ['adf_val', 'pval', 'usedlag', 'nobs', 'critical_vals', 'icbest', 'corr', 'd_val', 'max_len']

    return adf_df


def adjust_for_inflation(df, cpi_df, col):
    df_ = df[['Date', col]]
    cpi_df_ = cpi_df[['observation_date', 'Daily_inflation']]
    df_['YearMonth'] = df_['Date'].dt.to_period('M')
    cpi_df_['YearMonth'] = cpi_df_['observation_date'].dt.to_period('M')

    df_ = df_.merge(cpi_df_[['YearMonth', 'Daily_inflation']],
                    on='YearMonth', how='left')

    df['Close'] = df_['Close'] / df_['Daily_inflation']

    return df


def plot_adf_test(adf_data):
    d = adf_data['d_val']
    conf = adf_data.loc[0, 'critical_vals']['5%']

    plot_data = adf_data[['d_val', 'pval', 'max_len']]

    plt_ = plt
    plt_.figure(figsize=(8, 6))
    fig, ax1 = plt_.subplots()
    ax1.plot(d, adf_data['corr'], label='Corr', color='blue')

    ax2 = ax1.twinx()
    bbox = ax1.get_position()
    ax2.set_position([bbox.x0, bbox.y0, bbox.width, bbox.height])
    ax2.plot(d, adf_data['adf_val'], label='adf-test (right)', color='darkred')
    ax2.axhline(conf, label=f'Confidence Min {conf: .3f}', color='black', linestyle='--')
    ax2.axhline(conf, label=f'Correlation Min {0.9}', color='blue', linestyle='-.')

    # Add titles and labels
    plt_.title('ADF Test')
    plt_.xlabel('d-val')

    ax1.legend(loc='upper center')
    ax2.legend(loc='upper right')

    plot_labels = plot_data.columns
    plot_data = plot_data.round(3).values

    plt_.table(cellText=plot_data,
               colLabels=plot_labels,
               loc='center',
               bbox=[1.2, 0.1, 0.4, 0.8])  # [x, y, width, height]

    plt_.subplots_adjust(right=0.75)

    plt_.grid(True)
    plt_.tight_layout()

    return plt_


def plot_frac_FFD(test_arrs, df, save_loc):
    arr = df['Close'].values
    dates = pd.to_datetime(df['Date']).values
    for data_array in test_arrs:
        d = data_array[0]
        thres = data_array[1]
        w_df = data_array[2]

        valid_idx = len(dates) - len(w_df)
        x = dates[valid_idx:]
        y1 = arr[valid_idx:]
        y2 = w_df
        y2_avg = np.nanmean(y2)

        fig, ax1 = plt.subplots()
        ax1.plot(x, y1, 'b-', label='Vol - Real')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Vol - Real', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.spines['right'].set_position(('outward', 60))

        ax2.plot(x, y2, 'r-', label='Vol - FracDiff')
        ax2.axhline(y2_avg, color='green', linestyle='--', label=f'y2 avg = {y2_avg:.2f}')
        ax2.set_ylabel('Vol - FracDiff', color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        ax2.legend(loc='upper right')

        plt.title(f'd: {d} thres: {thres}')
        plt.tight_layout()
        save_name = f'{save_loc}\\diff_graph_d{d: .1f}_thres{thres}.png'
        plt.savefig(save_name)
        plt.close()


def main():
    data_loc = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data'
    data_end = 'NQ_daily_20240505_20040401.txt'
    cpi_file = f'{data_loc}\\inflation_adjusted\\CPIAUCSL_seasonally_adj.xlsx'
    remove_cols = ['Vol.1', 'OI', 'Time', 'AvgExp12', 'AvgExp24', 'Bearish_Double_Candle',
                   'Bullish_Double_Candle', 'VolAvg']
    test_col = 'Close'
    save_loc = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR\NQ\15min\15min_test_20years'
    save_loc = f'{save_loc}\\working folder\\fraction_diff_{test_col}'
    os.makedirs(save_loc, exist_ok=True)

    cpi_df = pd.read_excel(cpi_file)
    cpi_df['observation_date'] = pd.to_datetime(cpi_df['observation_date'])

    df = pd.read_csv(f'{data_loc}\\{data_end}')
    df_cols = df.columns
    df_cols = [col for col in df_cols if col not in remove_cols]
    df = df[df_cols]
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df[df['Date'] >= pd.to_datetime('2010-04-01', format='%Y-%m-%d')]

    df = adj_columns(df)
    df = adjust_for_inflation(df, cpi_df, test_col)
    arr = df[test_col].values

    test_size = 21

    thresholds = [.005, .001, .0005, .0001, .00005, .00001, .000005, .000001, .0000005, .0000001]
    print(thresholds)
    for thres in thresholds:
        print(thres)
        adf_data, test_arrs = adf_d_FFD_matrix(arr, test_size, log_scale=True, thres=thres)
        adf_data = organize_adf_data(adf_data)
        plot_frac_FFD(test_arrs, df, save_loc)
        plt = plot_adf_test(adf_data)

        save_name = f'{save_loc}\\{test_col}_d_analysis{thres: .6f}.png'
        print(save_name)
        plt.savefig(save_name)
        # plt.show()
        plt.close()


if __name__ == '__main__':
    main()