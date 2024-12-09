import numpy as np
import pandas as pd
import data_tools.math_tools2 as mt
import data_tools.general_tools as gt
import data_tools.data_trade_tools as tdt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from datetime import datetime, timedelta
import warnings
import tensorflow as tf

warnings.simplefilter(action='ignore', category=FutureWarning)


class MktDataHandler:
    def __init__(self, data_params):
        self.data_params = data_params
        self.all_secs = [self.data_params.security] + self.data_params.other_securities
        self.trade_data = tdt.TradeData(data_params)
        self.lstm_model = None

        self.dailydata = None
        self.intradata = None
        self.intradata = None
        self.security_df = None
        self.daily_temp_df = None
        self.ffd_df = None

        self.intra_scaler = None
        self.daily_scaler = None

        self.x_train_daily = None
        self.x_test_daily = None

        self.x_train_intra = None
        self.x_test_intra = None

        self.y_train_pnl_df = None
        self.y_train_wl_df = None
        self.y_test_pnl_df = None
        self.y_test_wl_df = None

        self.y_pnl_scaler = None
        self.y_wl_onehot_scaler = None
        self.intra_len = None

    def set_trade_data(self, trade_data):
        self.trade_data = trade_data
        # self.set_daily_time_len()

    def set_daily_time_len(self):
        """Move to lstm?"""
        open_time = datetime.strptime(f'{self.data_params.start_hour}:00', '%H:%M')
        close_time = datetime.strptime('15:00', '%H:%M')
        time_diff = close_time - open_time
        time_interval = int(''.join(filter(str.isdigit, self.data_params.time_frame_test)))
        self.intra_len = int(time_diff/timedelta(minutes=time_interval))

    def load_ffd_table(self):
        self.ffd_df = pd.read_excel(f'{self.data_params.strat_dat_loc}\\all_FFD_params.xlsx')


    def load_prep_daily_data(self):
        data_loc = f'{self.data_params.data_loc}'
        data_end = 'daily_20240505_20040401.txt'
        remove_cols = ['Vol.1', 'OI', 'Time', 'AvgExp12', 'AvgExp24', 'Bearish_Double_Candle',
                       'Bullish_Double_Candle', 'VolAvg']
        print('\nLoading Daily Data')

        dfs = []
        temp_dfs = []
        for sec in self.all_secs:
            print(f'...{sec}')
            temp_df = pd.read_csv(f'{data_loc}\\{sec}_{data_end}')
            temp_df = gt.convert_date_time(temp_df)
            daily_temp = temp_df[['DateTime', 'Close', 'Vol', 'OpenInt']]
            daily_temp.rename(columns={'Close': f'{sec}_Close_Daily',
                                       'Vol': f'{sec}_Vol_Daily',
                                       'OpenInt': f'{sec}_OpenInt_Daily'}, inplace=True)
            temp_dfs.append(daily_temp)

            cols = ['DateTime'] + [col for col in temp_df.columns.to_list()[:-1]]
            temp_df = temp_df[cols]

            cols_remove = [col for col in temp_df.columns if col in remove_cols]
            temp_df.drop(columns=cols_remove, inplace=True)

            for col in temp_df.columns[1:]:
                temp_df[col] = temp_df[col].astype(np.float32)
                temp_df.rename(columns={col: f'{sec}_{col}'}, inplace=True)

            temp_df = temp_df.sort_values(by='DateTime')
            temp_df = mt.set_various_data(temp_df, sec, 8)
            temp_df = mt.set_various_data(temp_df, sec, 24)
            temp_df = mt.add_high_low_diff(temp_df, sec)
            temp_df = mt.create_rsi(temp_df, sec)
            temp_df = mt.smooth_vol_oi_daily(temp_df, sec)

            dfs.append(temp_df)

        working_df = dfs[0]
        daily_temp_merged = temp_dfs[0]
        for df in dfs[1:]:
            working_df = pd.merge(working_df, df, on='DateTime')
        for df in temp_dfs[1:]:
            daily_temp_merged = pd.merge(daily_temp_merged, df, on='DateTime', how='outer')

        self.daily_temp_df = daily_temp_merged
        # working_df['Year'] = pd.to_datetime(working_df['DateTime']).dt.year
        working_df['Month'] = pd.to_datetime(working_df['DateTime']).dt.month
        working_df['Day'] = pd.to_datetime(working_df['DateTime']).dt.dayofweek

        working_df = mt.encode_time_features(working_df, intra=False)

        self.dailydata = working_df
        self.finish_data_prep()

    def load_prep_intra_data(self):
        data_loc = f'{self.data_params.data_loc}'

        data_end = f'{self.data_params.time_frame_train}_20240505_20040401.txt'
        remove_cols = ['Vol.1', 'OI', 'Time', 'AvgExp12', 'AvgExp24', 'Bearish_Double_Candle',
                       'Bullish_Double_Candle', 'Date', 'Time', 'Up', 'Down', 'VolAvg']
        print('\nLoading Intraday Data')

        dfs = []
        for sec in self.all_secs:
            print(f'...{sec}')
            temp_df = pd.read_csv(f'{data_loc}\\{sec}_{data_end}')
            temp_df = gt.convert_date_time(temp_df)
            temp_df = temp_df[temp_df['DateTime'] >= pd.to_datetime(self.data_params.start_train_date) -
                              timedelta(days=90)].reset_index(drop=True)
            if sec == self.data_params.security:
                self.security_df = temp_df[['DateTime', 'Close']]

            cols = ['DateTime'] + [col for col in temp_df.columns.to_list()[:-1]]
            temp_df = temp_df[cols]

            cols_remove = [col for col in temp_df.columns if col in remove_cols]
            temp_df.drop(columns=cols_remove, inplace=True)

            for col in temp_df.columns[1:]:
                temp_df[col] = temp_df[col].astype(np.float32)
                temp_df.rename(columns={col: f'{sec}_{col}'}, inplace=True)

            temp_df = temp_df.sort_values(by='DateTime')

            temp_df = mt.set_various_data(temp_df, sec, 8)
            temp_df = mt.set_various_data(temp_df, sec, 24)
            temp_df = mt.add_high_low_diff(temp_df, sec)
            temp_df = mt.create_rsi(temp_df, sec)
            temp_df = mt.smooth_vol_oi_intra(temp_df, self.daily_temp_df, self.all_secs)

            temp_df = mt.ffd_scale_ohlc(temp_df, self.ffd_df)

            dfs.append(temp_df)

        working_df = dfs[0]
        for df in dfs[1:]:
            working_df = pd.merge(working_df, df, on='DateTime')

        working_df = working_df[working_df['DateTime'] >= pd.to_datetime(self.data_params.start_train_date)]
        # working_df['Year'] = pd.to_datetime(working_df['DateTime']).dt.year
        working_df['Month'] = pd.to_datetime(working_df['DateTime']).dt.month
        working_df['Day'] = pd.to_datetime(working_df['DateTime']).dt.dayofweek
        working_df['Hour'] = pd.to_datetime(working_df['DateTime']).dt.hour
        working_df['Minute'] = pd.to_datetime(working_df['DateTime']).dt.minute

        working_df = mt.encode_time_features(working_df, intra=True)

        self.intradata = working_df

        self.finish_data_prep(daily=False)

    def finish_data_prep(self, daily=True):
        print('\nFinishing Intraday Data Prep')
        if daily:
            df = self.dailydata
        else:
            df = self.intradata

        df = gt.sort_data_cols(df)
        df = gt.fill_na_inf(df)

        for sec in self.all_secs:
            df.drop(columns=[f'{sec}_Vol', f'{sec}_OpenInt'], inplace=True)

        if daily:
            self.dailydata = df
        else:
            self.intradata = df

    def inf_check(self):
        self.dailydata = gt.fill_na_inf(self.dailydata)
        self.intradata = gt.fill_na_inf(self.intradata)

    def set_x_train_test_datasets(self):
        print('\nBuilding X-Train and Test Datasets')
        self.x_train_intra = self.intradata[self.intradata['DateTime'].dt.date.isin(self.trade_data.train_dates)]
        self.x_test_intra = self.intradata[self.intradata['DateTime'].dt.date.isin(self.trade_data.test_dates)]

        train_dates = (
                self.trade_data.add_to_daily_dates(self.lstm_model.daily_len, train=True) + self.trade_data.train_dates)
        self.x_train_daily = self.dailydata[self.dailydata['DateTime'].dt.date.isin(train_dates)]

        test_dates = (
                self.trade_data.add_to_daily_dates(self.lstm_model.daily_len, train=False) + self.trade_data.test_dates)
        self.x_test_daily = self.dailydata[self.dailydata['DateTime'].dt.date.isin(test_dates)]

    def scale_x_data(self, load_scalers, test_date, traintf, i):
        print('\nScaling X Data')
        self.x_train_intra = gt.arrange_xcols_for_scaling(self.x_train_intra)
        self.x_test_intra = gt.arrange_xcols_for_scaling(self.x_test_intra)
        self.x_train_daily = gt.arrange_xcols_for_scaling(self.x_train_daily)
        self.x_test_daily = gt.arrange_xcols_for_scaling(self.x_test_daily)

        self.x_train_intra.iloc[:, 1:] = self.x_train_intra.iloc[:, 1:].astype('float32')
        self.x_test_intra.iloc[:, 1:] = self.x_test_intra.iloc[:, 1:].astype('float32')
        self.x_train_daily.iloc[:, 1:] = self.x_train_daily.iloc[:, 1:].astype('float32')
        self.x_test_daily.iloc[:, 1:] = self.x_test_daily.iloc[:, 1:].astype('float32')

        if traintf:
            if load_scalers:
                print('Loading Previous X-Scalers')
                # x_train_fit_intra = self.prep_xy_fit_data(self.x_train_intra, test_date)
                # self.intra_scaler.partial_fit(x_train_fit_intra.iloc[:, 1:].values)
                #
                # x_train_fit_daily = self.prep_xy_fit_data(self.x_train_daily, test_date)
                # self.daily_scaler.partial_fit(x_train_fit_daily.iloc[:, 1:].values)

            else:
                print('Creating New X-Scalers')
                self.intra_scaler = StandardScaler()
                self.intra_scaler.fit(self.x_train_intra.iloc[:, 1:].values)

                self.daily_scaler = StandardScaler()
                self.daily_scaler.fit(self.x_train_daily.iloc[:, 1:].values)

        self.x_train_intra.iloc[:, 1:] = self.intra_scaler.transform(self.x_train_intra.iloc[:, 1:].values)
        self.x_test_intra.iloc[:, 1:] = self.intra_scaler.transform(self.x_test_intra.iloc[:, 1:].values)

        self.x_train_daily.iloc[:, 1:] = self.daily_scaler.transform(self.x_train_daily.iloc[:, 1:].values)
        self.x_test_daily.iloc[:, 1:] = self.daily_scaler.transform(self.x_test_daily.iloc[:, 1:].values)

    def prep_xy_fit_data(self, df, test_date):
        last_test_date = pd.to_datetime(test_date) - timedelta(days=self.data_params.test_period_days)
        last_last_test_date = last_test_date - timedelta(days=self.data_params.test_period_days)
        x_train_fit_data = df[(df['DateTime'] > last_last_test_date) &
                              (df['DateTime'] <= last_test_date)]

        return x_train_fit_data

    def scale_y_pnl_data(self, load_scalers, test_date, traintf, i):
        print('\nScaling Y-pnl Data')

        self.y_train_pnl_df = self.trade_data.y_train_df.iloc[:, :2]
        self.y_train_pnl_df.iloc[:, 1] = self.y_train_pnl_df.iloc[:, 1].astype('float32')

        self.y_test_pnl_df = self.trade_data.y_test_df.iloc[:, :2]
        self.y_test_pnl_df.iloc[:, 1] = self.y_test_pnl_df.iloc[:, 1].astype('float32')

        if traintf:
            if load_scalers:
                print('Loading Previous PnL-Scalers')
                # y_train_train_fit = self.prep_xy_fit_data(self.y_train_pnl_df, test_date)
                # if len(y_train_train_fit) > 0:
                #     self.y_pnl_scaler.partial_fit(y_train_train_fit.iloc[:, 1].values.reshape(-1, 1))

            else:
                print('Creating New PnL-Scalers')
                self.y_pnl_scaler = StandardScaler()
                self.y_pnl_scaler.fit(self.y_train_pnl_df.iloc[:, 1].values.reshape(-1, 1))

        pnl_scaled = self.y_pnl_scaler.transform(self.y_train_pnl_df.iloc[:, 1].values.reshape(-1, 1))
        self.y_train_pnl_df.iloc[:, 1] = pnl_scaled.reshape(-1, 1)

        if len(self.y_test_pnl_df) > 0:
            pnl_scaled = self.y_pnl_scaler.transform(self.y_test_pnl_df.iloc[:, 1].values.reshape(-1, 1))
            self.y_test_pnl_df.iloc[:, 1] = pnl_scaled.reshape(-1, 1)

    def onehot_y_wl_data(self):
        print('\nOnehotting WL Data')
        self.y_wl_onehot_scaler = OneHotEncoder(sparse_output=False)

        self.y_train_wl_df = self.trade_data.y_train_df.iloc[:, [0, 2]]

        wl_dat = self.y_train_wl_df.iloc[:, 1].values
        wl_dat = self.y_wl_onehot_scaler.fit_transform(wl_dat.reshape(-1, 1))

        self.y_train_wl_df[['Loss', 'Win']] = wl_dat
        self.y_train_wl_df.drop('Win_Loss', inplace=True, axis=1)

        self.y_test_wl_df = self.trade_data.y_test_df.iloc[:, [0, 2]]
        if len(self.y_test_wl_df) > 0:
            wl_dat = self.y_test_wl_df.iloc[:, 1].values
            wl_dat = self.y_wl_onehot_scaler.transform(wl_dat.reshape(-1, 1))
            self.y_test_wl_df[['Loss', 'Win']] = wl_dat
            self.y_test_wl_df.drop('Win_Loss', inplace=True, axis=1)

    def grab_prep_trade(self, y_pnl_df, y_wl_df, x_intraday, train_ind, daily_len):
        while True:
            try:
                trade_dt = y_pnl_df.iloc[train_ind]['DateTime']

                x_daily_input = self.dailydata[self.dailydata['DateTime'] <
                                               trade_dt].tail(daily_len).values[:, 1:]
                x_daily_input = gt.pad_to_length(x_daily_input, daily_len)

                x_intra_input = x_intraday[(x_intraday['DateTime'] <= trade_dt) &
                                           (x_intraday['DateTime'] >=
                                            trade_dt.replace(hour=self.data_params.start_hour,
                                                             minute=self.data_params.start_minute))]

                x_intra_input = x_intra_input.tail(self.intra_len).values[:, 1:]
                x_intra_input = gt.pad_to_length(x_intra_input, self.intra_len)

                y_pnl_input = np.array([y_pnl_df.iloc[train_ind, 1]])

                y_wl_input = y_wl_df.iloc[train_ind, 1:].values

                yield x_daily_input, x_intra_input, y_pnl_input, y_wl_input

            except StopIteration:
                break

    def create_batch_input(self, train_inds, daily_len, train=True):
        while True:
            y_pnl_df = self.y_train_pnl_df if train else self.y_test_pnl_df
            y_wl_df = self.y_train_wl_df if train else self.y_test_wl_df
            x_intraday = self.x_train_intra if train else self.x_test_intra

            x_day_arr, x_intra_arr, y_pnl_arr, y_wl_arr = [], [], [], []

            try:
                for train_ind in train_inds:
                    x_day, x_intra, y_pnl, y_wl = next(self.grab_prep_trade(y_pnl_df, y_wl_df, x_intraday,
                                                                            train_ind, daily_len))
                    x_day_arr.append(x_day)
                    x_intra_arr.append(x_intra)
                    y_pnl_arr.append(y_pnl)
                    y_wl_arr.append(y_wl)

                x_day_arr = tf.convert_to_tensor(x_day_arr, dtype=tf.float32)
                x_intra_arr = tf.convert_to_tensor(x_intra_arr, dtype=tf.float32)
                y_pnl_arr = tf.convert_to_tensor(y_pnl_arr, dtype=tf.float32)
                y_wl_arr = tf.convert_to_tensor(y_wl_arr, dtype=tf.float32)

                # yield x_day_arr, x_intra_arr, y_pnl_arr, y_wl_arr
                yield (x_day_arr, x_intra_arr), {'wl_class': y_wl_arr, 'pnl': y_pnl_arr}

            except StopIteration:
                if x_day_arr:
                    x_day_arr = tf.convert_to_tensor(x_day_arr, dtype=tf.float32)
                    x_intra_arr = tf.convert_to_tensor(x_intra_arr, dtype=tf.float32)
                    y_pnl_arr = tf.convert_to_tensor(y_pnl_arr, dtype=tf.float32)
                    y_wl_arr = tf.convert_to_tensor(y_wl_arr, dtype=tf.float32)
                # yield x_day_arr, x_intra_arr, y_pnl_arr, y_wl_arr
                yield (x_day_arr, x_intra_arr), {'wl_class': y_wl_arr, 'pnl': y_pnl_arr}

            break

    def subset_start_time(self):
        start_time = pd.Timestamp(f'{self.data_params.start_hour:02}:{self.data_params.start_minute:02}:00').time()

        self.x_train_intra['time'] = self.x_train_intra['DateTime'].dt.time
        subset_df = self.x_train_intra[
            (self.x_train_intra['time'] >= start_time) &
            (self.x_train_intra['time'] <= pd.Timestamp('16:00:00').time())]
        self.x_train_intra = subset_df.drop(columns=['time'])

        self.x_test_intra['time'] = self.x_test_intra['DateTime'].dt.time
        subset_df = self.x_test_intra[
            (self.x_test_intra['time'] >= start_time) &
            (self.x_test_intra['time'] <= pd.Timestamp('16:00:00').time())]
        self.x_test_intra = subset_df.drop(columns=['time'])


