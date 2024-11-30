import numpy as np
import pandas as pd
import old_files.math_tools as mt
import data_tools.general_tools as gt
import data_tools.data_trade_tools as tdt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class MktDataHandler:
    def __init__(self, data_params):
        self.data_params = data_params
        self.all_secs = [self.data_params.security] + self.data_params.other_securities
        self.trade_data = tdt.TradeData(data_params)

        self.dailydata = None
        self.intradata = None
        self.security_df = None

        self.intra_scaler = None
        self.month_onehot_scaler = None
        self.day_onehot_scaler = None

        self.daily_train_test = None

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
        self.set_daily_time_len()

    def set_daily_time_len(self):
        """Move to lstm?"""
        open_time = datetime.strptime(f'{self.data_params.start_hour}:00', '%H:%M')
        close_time = datetime.strptime('15:00', '%H:%M')
        time_diff = close_time - open_time
        time_interval = int(''.join(filter(str.isdigit, self.data_params.time_frame)))
        self.intra_len = int(time_diff/timedelta(minutes=time_interval))

    def load_prep_data(self, daily=True):
        data_loc = f'{self.data_params.data_loc}'
        if daily:
            data_end = 'daily_20240505_20040401.txt'
            remove_cols = ['Vol.1', 'OI', 'Time', 'AvgExp12', 'AvgExp24', 'Bearish_Double_Candle',
                           'Bullish_Double_Candle', 'DateTime', 'VolAvg']
            date_or_dt = 'Date'
            print('\nLoading Daily Data')

        else:
            data_end = f'{self.data_params.time_frame}_20240505_20040401.txt'
            remove_cols = ['Vol.1', 'OI', 'Time', 'AvgExp12', 'AvgExp24', 'Bearish_Double_Candle',
                           'Bullish_Double_Candle', 'Date', 'Time', 'Up', 'Down', 'VolAvg']
            date_or_dt = 'DateTime'
            print('\nLoading Intraday Data')

        dfs = []
        for sec in self.all_secs:
            print(f'...{sec}')
            temp_df = pd.read_csv(f'{data_loc}\\{sec}_{data_end}')

            if (sec == self.data_params.security) and not daily:
                self.security_df = temp_df[['Date', 'Time', 'Close']]
                self.security_df = gt.convert_date_time(self.security_df)
                self.security_df = self.security_df[['DateTime', 'Close']]

            temp_df = mt.set_various_data(temp_df)
            temp_df = gt.convert_date_time(temp_df)
            temp_df = temp_df.sort_values(by='DateTime')

            cols_remove = [col for col in temp_df.columns if col in remove_cols]
            temp_df.drop(columns=cols_remove, inplace=True)

            if not daily:
                cols = ['DateTime'] + temp_df.columns[:-1].to_list()
                temp_df = temp_df[cols]

            for col in temp_df.columns[1:]:
                temp_df[col] = temp_df[col].astype(np.float32)
                temp_df.rename(columns={col: f'{sec}_{col}'}, inplace=True)

            dfs.append(temp_df)

        working_df = dfs[0]
        for df in dfs[1:]:
            working_df = pd.merge(working_df, df, on=date_or_dt)

        working_df[date_or_dt] = pd.to_datetime(working_df[date_or_dt])
        working_df = working_df[working_df[date_or_dt] >= pd.to_datetime(self.data_params.start_train_date)]
        working_df['Month'] = pd.to_datetime(working_df[date_or_dt]).dt.month - 6
        working_df['Day'] = pd.to_datetime(working_df[date_or_dt]).dt.dayofweek - 2

        if daily:
            working_df['Date'] = working_df['Date'].dt.date
            self.dailydata = working_df
        else:
            self.intradata = working_df

        self.finish_data_prep(daily)

    def finish_data_prep(self, daily=True):
        print('\nFinishing Intraday Data Prep')
        if daily:
            df = self.dailydata
        else:
            df = self.intradata

        df = mt.create_rsi(df, self.all_secs)
        df = mt.add_high_low_diff(df,
                                  self.data_params.other_securities,
                                  self.data_params.security)

        df = mt.smooth_vol_oi(df, self.all_secs)
        df = mt.scale_open_close(df)
        df = gt.sort_data_cols(df)
        df = gt.fill_na_inf(df)

        for sec in self.all_secs:
            df.drop(columns=f'{sec}_Vol', inplace=True)

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

    def scale_x_data(self, load_scalers, test_date):
        print('\nScaling X Data')
        # self.intra_scaler = RobustScaler(quantile_range=(3, 97))
        self.x_train_intra = gt.arrange_xcols_for_scaling(self.x_train_intra)
        self.x_test_intra = gt.arrange_xcols_for_scaling(self.x_test_intra)

        self.x_train_intra.iloc[:, 1:] = self.x_train_intra.iloc[:, 1:].astype('float32')

        if load_scalers:
            print('Loading Previous Scalers')
            last_test_date = pd.to_datetime(test_date) - timedelta(days=7)
            last_last_test_date = last_test_date - timedelta(days=7)
            x_train_fit_data = self.x_train_intra[(self.x_train_intra['DateTime'] > last_last_test_date) &
                                                  (self.x_train_intra['DateTime'] <= last_test_date)]
            self.intra_scaler.partial_fit(x_train_fit_data.iloc[:, 1:].values)
            # self.intra_scaler.fit(x_train_fit_data.iloc[:, 1:].values)

        else:
            self.intra_scaler = StandardScaler()
            self.intra_scaler.partial_fit(self.x_train_intra.iloc[:, 1:].values)
            # self.intra_scaler.fit(self.x_train_intra.iloc[:, 1:].values)

        self.x_train_intra.iloc[:, 1:] = self.intra_scaler.transform(self.x_train_intra.iloc[:, 1:].values)
        self.x_test_intra.iloc[:, 1:] = self.x_test_intra.iloc[:, 1:].astype('float32')
        self.x_test_intra.iloc[:, 1:] = self.intra_scaler.transform(self.x_test_intra.iloc[:, 1:].values)

        self.daily_train_test = gt.arrange_xcols_for_scaling(self.dailydata)
        self.daily_train_test.iloc[:, 1:] = self.daily_train_test.iloc[:, 1:].astype('float32')
        self.daily_train_test.iloc[:, 1:] = self.intra_scaler.transform(
            self.daily_train_test.iloc[:, 1:].values)

    def onehot_month_day(self):
        self.day_onehot_scaler = OneHotEncoder(sparse_output=False)
        self.day_onehot_scaler.fit(list(range(0, 5)))
        oh_day_train = self.day_onehot_scaler.transform(self.x_train_intra.iloc[:, -2].values.reshape(-1, 1))
        oh_day_test = self.day_onehot_scaler.transform(self.x_test_intra.iloc[:, -2].values.reshape(-1, 1))
        oh_day_daily = self.day_onehot_scaler.transform(self.dailydata.iloc[:, -2].values.reshape(-1, 1))

        self.month_onehot_scaler = OneHotEncoder(sparse_output=False)
        self.month_onehot_scaler.fit(list(range(1, 13)))
        oh_month_train = self.day_onehot_scaler.transform(self.x_train_intra.iloc[:, -1].values.reshape(-1, 1))
        oh_month_test = self.day_onehot_scaler.transform(self.x_test_intra.iloc[:, -1].values.reshape(-1, 1))
        oh_month_daily = self.day_onehot_scaler.transform(self.dailydata.iloc[:, -1].values.reshape(-1, 1))

        self.x_train_intra = self.x_train_intra.iloc[:, :-2]
        self.x_test_intra = self.x_test_intra.iloc[:, :-2]
        self.dailydata = self.dailydata.iloc[:, :-2]

        self.x_train_intra = gt.add_arr_to_df(self.x_train_intra, oh_day_train)
        self.x_test_intra = gt.add_arr_to_df(self.x_test_intra, oh_day_test)
        self.dailydata = gt.add_arr_to_df(self.dailydata, oh_day_daily)

        self.x_train_intra = gt.add_arr_to_df(self.x_train_intra, oh_month_train)
        self.x_test_intra = gt.add_arr_to_df(self.x_test_intra, oh_month_test)
        self.dailydata = gt.add_arr_to_df(self.dailydata, oh_month_daily)

    def scale_y_pnl_data(self, load_scalers, test_date):
        print('\nScaling Y-pnl Data')

        self.y_train_pnl_df = self.trade_data.y_train_df.iloc[:, :2]
        self.y_train_pnl_df.iloc[:, 1] = self.y_train_pnl_df.iloc[:, 1].astype('float32')

        if load_scalers:
            last_test_date = pd.to_datetime(test_date) - timedelta(days=7)
            last_last_test_date = last_test_date - timedelta(days=7)
            y_train_train_fit = self.y_train_pnl_df[(self.y_train_pnl_df['DateTime'] > last_last_test_date) &
                                                    (self.y_train_pnl_df['DateTime'] <= last_test_date)]
            self.y_pnl_scaler.partial_fit(y_train_train_fit.iloc[:, 1].values.reshape(-1, 1))
            # self.y_pnl_scaler.fit(y_train_train_fit.iloc[:, 1].values.reshape(-1, 1))

        else:
            self.y_pnl_scaler = StandardScaler()
            self.y_pnl_scaler.partial_fit(self.y_train_pnl_df.iloc[:, 1].values.reshape(-1, 1))

        # self.y_pnl_scaler = RobustScaler(quantile_range=(5, 95))
        # self.y_pnl_scaler.fit(self.y_train_pnl_df.iloc[:, 1].values.reshape(-1, 1))

        pnl_scaled = self.y_pnl_scaler.transform(self.y_train_pnl_df.iloc[:, 1].values.reshape(-1, 1))
        self.y_train_pnl_df.iloc[:, 1] = pnl_scaled.reshape(-1, 1)

        self.y_test_pnl_df = self.trade_data.y_test_df.iloc[:, :2]
        self.y_test_pnl_df.iloc[:, 1] = self.y_test_pnl_df.iloc[:, 1].astype('float32')

        pnl_scaled = self.y_pnl_scaler.transform(self.y_test_pnl_df.iloc[:, 1].values.reshape(-1, 1))
        self.y_test_pnl_df.iloc[:, 1] = pnl_scaled.reshape(-1, 1)

    def onehot_y_wl_data(self):
        print('\nOnehotting WL Data')
        self.y_wl_onehot_scaler = OneHotEncoder(sparse_output=False)

        self.y_train_wl_df = self.trade_data.y_train_df.iloc[:, [0, 2]]
        wl_dat = self.y_train_wl_df.iloc[:, 1].values
        wl_dat = self.y_wl_onehot_scaler.fit_transform(wl_dat.reshape(-1, 1))

        self.y_train_wl_df['Loss'] = wl_dat[:, 0]
        self.y_train_wl_df['Win'] = wl_dat[:, 1]
        self.y_train_wl_df.drop('Win_Loss', inplace=True, axis=1)

        self.y_test_wl_df = self.trade_data.y_test_df.iloc[:, [0, 2]]
        wl_dat = self.y_test_wl_df.iloc[:, 1].values
        wl_dat = self.y_wl_onehot_scaler.transform(wl_dat.reshape(-1, 1))

        self.y_test_wl_df['Loss'] = wl_dat[:, 0]
        self.y_test_wl_df['Win'] = wl_dat[:, 1]
        self.y_test_wl_df.drop('Win_Loss', inplace=True, axis=1)

    def grab_prep_trade(self, y_pnl_df, y_wl_df, x_intraday, train_ind, daily_len):
        while True:
            try:
                trade_dt = y_pnl_df.loc[train_ind, 'DateTime']

                x_daily_input = self.daily_train_test[self.daily_train_test['Date'] <
                                                      trade_dt.date()].tail(daily_len).values[:, 1:]
                x_daily_input = gt.pad_to_length(x_daily_input, daily_len)

                x_intra_input = x_intraday[(x_intraday['DateTime'] <= trade_dt) &
                                           (x_intraday['DateTime'] >=
                                            trade_dt.replace(hour=self.data_params.start_hour,
                                                             minute=self.data_params.start_minute))]

                x_intra_input = x_intra_input.tail(self.intra_len).values[:, 1:]
                x_intra_input = gt.pad_to_length(x_intra_input, self.intra_len)

                y_pnl_input = np.array([y_pnl_df[y_pnl_df['DateTime'] == trade_dt].values[0, 1]])

                y_wl_input = y_wl_df[y_wl_df['DateTime'] == trade_dt].values[0, 1:]

                yield x_daily_input, x_intra_input, y_pnl_input, y_wl_input

            except StopIteration:
                break

    def create_batch_input(self, train_inds, daily_len, train=True):
        while True:
            if train:
                y_pnl_df = self.y_train_pnl_df
                y_wl_df = self.y_train_wl_df
                x_intraday = self.x_train_intra
            else:
                y_pnl_df = self.y_test_pnl_df
                y_wl_df = self.y_test_wl_df
                x_intraday = self.x_test_intra

            x_day_arr, x_intra_arr, y_pnl_arr, y_wl_arr = [], [], [], []

            try:
                for train_ind in train_inds:
                    x_day, x_intra, y_pnl, y_wl = next(self.grab_prep_trade(y_pnl_df, y_wl_df, x_intraday,
                                                                            train_ind, daily_len))
                    x_day_arr.append(x_day)
                    x_intra_arr.append(x_intra)
                    y_pnl_arr.append(y_pnl)
                    y_wl_arr.append(y_wl)

                x_day_arr = np.array(x_day_arr).astype(np.float32)
                x_intra_arr = np.array(x_intra_arr).astype(np.float32)
                y_pnl_arr = np.array(y_pnl_arr).astype(np.float32)
                y_wl_arr = np.array(y_wl_arr).astype(np.float32)

                yield x_day_arr, x_intra_arr, y_pnl_arr, y_wl_arr

                x_day_arr, x_intra_arr, y_pnl_arr, y_wl_arr = [], [], [], []

            except StopIteration:
                if x_day_arr:
                    x_day_arr = np.array(x_day_arr).astype(np.float32)
                    x_intra_arr = np.array(x_intra_arr).astype(np.float32)
                    y_pnl_arr = np.array(y_pnl_arr).astype(np.float32)
                    y_wl_arr = np.array(y_wl_arr).astype(np.float32)

                yield x_day_arr, x_intra_arr, y_pnl_arr, y_wl_arr

            break
