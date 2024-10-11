import numpy as np
import random
import copy
import pandas as pd
import neural_network.nn_tools.math_tools as mt
import gen_data_tools.general_tools as gt
import neural_network.nn_tools.lstm_trade_data_tools as tdt
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from datetime import datetime, timedelta

# np.random.seed(42)


class LstmDataHandler:
    def __init__(self, data_param_dict, trade_data_dict):
        self.data_params = data_param_dict
        self.all_secs = [self.data_params['security']] + self.data_params['other_securities']
        self.trade_data = tdt.TradeData(trade_data_dict, data_param_dict)

        self.dailydata = None
        self.intradata = None

        self.intra_scaler = None
        self.daily_scaler = None

        self.x_train_intra = None
        self.x_test_intra = None

        self.y_train_pnl_df = None
        self.y_train_wl_df = None
        self.y_test_pnl_df = None
        self.y_test_wl_df = None

        self.y_pnl_scaler = None
        self.y_wl_onehot_scaler = None

        self.intra_len = 0

    def get_trade_data(self):
        self.trade_data.get_trade_data()
        self.set_daily_time_len()

    def set_daily_time_len(self):
        open_time = datetime.strptime(f'{self.data_params["start_hour"]}:00', '%H:%M')
        close_time = datetime.strptime('15:00', '%H:%M')
        time_diff = close_time - open_time
        time_interval = int(''.join(filter(str.isdigit, self.data_params['time_frame'])))
        self.intra_len = int(time_diff/timedelta(minutes=time_interval))

    def load_prep_daily_data(self):
        print('\nLoading Daily Data')
        daily_dfs = []
        for sec in [self.data_params['security']] + self.data_params['other_securities']:
            print(f'...{sec}')
            temp_df = pd.read_csv(f'{self.data_params["data_loc"]}\\{sec}_daily_20240505_20040401.txt')
            temp_df['ATR'] = mt.create_atr(temp_df, 8)
            temp_df = gt.convert_date_time(temp_df)
            temp_df = temp_df.sort_values(by='DateTime')
            for col in temp_df.columns:
                if col in ['Vol.1', 'OI', 'Time', 'AvgExp12', 'AvgExp24', 'Bearish_Double_Candle',
                           'Bullish_Double_Candle', 'DateTime']:
                    temp_df.drop(columns=[col], inplace=True)

            temp_df = mt.set_various_data(temp_df)

            for col in temp_df.columns[1:]:
                temp_df[col] = temp_df[col].astype(np.float32)
                temp_df.rename(columns={col: f'{sec}_{col}'}, inplace=True)

            daily_dfs.append(temp_df)

        self.dailydata = daily_dfs[0]
        for df in daily_dfs[1:]:
            self.dailydata = pd.merge(self.dailydata, df, on='Date')

        self.dailydata = (
            self.dailydata)[self.dailydata['Date'] >= pd.to_datetime(self.data_params['start_train_date']).date()]
        self.finish_data_prep()

    def load_prep_intra_data(self):
        print('\nLoading Daily Data')
        intra_dfs = []
        for sec in [self.data_params['security']] + self.data_params['other_securities']:
            print(f'...{sec}')
            temp_df = pd.read_csv(f'{self.data_params["data_loc"]}\\'
                                  f'{sec}_{self.data_params["time_frame"]}_20240505_20040401.txt')
            temp_df['ATR'] = mt.create_atr(temp_df, 8)
            temp_df = gt.convert_date_time(temp_df)
            temp_df = temp_df.sort_values(by='DateTime')

            for col in temp_df.columns:
                if col in ['Vol.1', 'OI', 'Time', 'AvgExp12', 'AvgExp24', 'Bearish_Double_Candle',
                           'Bullish_Double_Candle', 'Date', 'Time', 'Up', 'Down']:
                    temp_df.drop(columns=[col], inplace=True)

            temp_df = temp_df[['DateTime'] + temp_df.columns[:-1].to_list()]
            temp_df = mt.set_various_data(temp_df)

            for col in temp_df.columns[1:]:
                temp_df[col] = temp_df[col].astype(np.float32)
                temp_df.rename(columns={col: f'{sec}_{col}'}, inplace=True)

            intra_dfs.append(temp_df)

        self.intradata = intra_dfs[0]
        for df in intra_dfs[1:]:
            self.intradata = pd.merge(self.intradata, df, on='DateTime')

        self.intradata = (
            self.intradata)[self.intradata['DateTime'].dt.date >=
                            pd.to_datetime(self.data_params['start_train_date']).date()]
        self.finish_data_prep(daily=False)

    def finish_data_prep(self, daily=True):
        print('\nFinishing Intraday Data Prep')
        if daily:
            df = self.dailydata
        else:
            df = self.intradata

        df = mt.create_rsi(df, self.all_secs)
        df = mt.add_high_low_diff(df,
                                  self.data_params['other_securities'],
                                  self.data_params['security'])

        df = mt.smooth_vol_oi(df, self.all_secs)
        df = mt.scale_open_close(df)
        df = gt.sort_data_cols(df)
        df = gt.fill_na_inf(df)

        if daily:
            self.dailydata = df
        else:
            self.intradata = df

    def set_intraday_data(self):
        print('\nSetting Intraday Data')
        sec = self.data_params['security']

        self.intradata['Candle_Size_OC'] = ((self.intradata[f'{sec}_Close'] - self.intradata[f'{sec}_Open'])/
                                            self.intradata[f'{sec}_Close'])

        self.intradata['Candle_Ratio_OC'] = \
            ((self.intradata[f'{sec}_Close'] - self.intradata[f'{sec}_Open'])/
             (self.intradata[f'{sec}_Close'].shift(1) - self.intradata[f'{sec}_Open'].shift(1)))

        self.intradata['Candle_Size_HL'] = ((self.intradata[f'{sec}_High'] - self.intradata[f'{sec}_Low'])/
                                            self.intradata[f'{sec}_High'])

        self.intradata['Candle_Ratio_HL'] = \
            ((self.intradata[f'{sec}_High'] - self.intradata[f'{sec}_Low']) /
             (self.intradata[f'{sec}_High'].shift(1) - self.intradata[f'{sec}_Low'].shift(1)))

    def set_x_train_test_datasets(self):
        print('\nBuilding X-Train and Test Datasets')
        self.x_train_intra = self.intradata[self.intradata['DateTime'].dt.date.isin(self.trade_data.train_dates)]
        self.x_train_intra = gt.fill_na_inf(self.x_train_intra)

        self.x_test_intra = self.intradata[self.intradata['DateTime'].dt.date.isin(self.trade_data.test_dates)]
        self.x_test_intra = gt.fill_na_inf(self.x_test_intra)

    def scale_x_data(self):
        print('\nScaling X Data')
        self.intra_scaler = RobustScaler(quantile_range=(3, 97))

        self.x_train_intra.iloc[:, 1:] = (
            self.intra_scaler.fit_transform(self.x_train_intra.iloc[:, 1:].values.astype('float32')))
        self.x_test_intra.iloc[:, 1:] = (
            self.intra_scaler.transform(self.x_test_intra.iloc[:, 1:].values.astype('float32')))

        self.daily_scaler = RobustScaler()
        self.dailydata.iloc[:, 1:] = (
            self.intra_scaler.fit_transform(self.dailydata.iloc[:, 1:].values.astype('float32')))

    def scale_y_pnl_data(self):
        print('\nScaling Y-pnl Data')
        self.y_pnl_scaler = RobustScaler(quantile_range=(5, 95))

        self.y_train_pnl_df = self.trade_data.y_train_df.iloc[:, :2]
        pnl_scaled = (
            self.y_pnl_scaler.fit_transform(self.y_train_pnl_df.iloc[:, 1].values.astype('float32').reshape(-1, 1)))
        self.y_train_pnl_df.iloc[:, 1] = pnl_scaled.reshape(-1, 1)

        self.y_test_pnl_df = self.trade_data.y_test_df.iloc[:, :2]
        pnl_scaled = (
            self.y_pnl_scaler.transform(self.y_test_pnl_df.iloc[:, 1].values.astype('float32').reshape(-1, 1)))
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

    def grab_prep_trade(self, y_pnl_df, y_wl_df, x_intraday, train_ind, daily_len, intra_len):
        while True:
            try:
                trade_dt = y_pnl_df.loc[train_ind, 'DateTime']

                x_daily_input = self.dailydata[self.dailydata['Date'] < trade_dt.date()].tail(23).values[:, 1:]
                x_daily_input = gt.pad_to_length(x_daily_input, daily_len)

                x_intra_input = x_intraday[(x_intraday['DateTime'] <= trade_dt) &
                                           (x_intraday['DateTime'] >=
                                            trade_dt.replace(hour=self.data_params['start_hour'],
                                                             minute=self.data_params['start_minute']))].values[:, 1:]
                x_intra_input = gt.pad_to_length(x_intra_input, intra_len)

                y_pnl_input = np.array([y_pnl_df[y_pnl_df['DateTime'] == trade_dt].values[0, 1]])

                y_wl_input = y_wl_df[y_wl_df['DateTime'] == trade_dt].values[0, 1:]

                yield x_daily_input, x_intra_input, y_pnl_input, y_wl_input

            except StopIteration:
                break

    def create_batch_input(self, train_inds, daily_len, intra_len, train=True):
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
                                                                            train_ind, daily_len, intra_len))
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
