import numpy as np
import random
import pandas as pd
import data_tools.general_tools as gt
from datetime import datetime, timedelta


class TradeData:
    def __init__(self, data_params):
        self.data_params = data_params
        self.data_loc = str
        self.trade_df = None
        self.param_df = None

        self.working_df = None
        self.y_train_df = None

        self.train_dates = []
        self.curr_test_date = None
        self.test_dates = []
        self.y_test_df = None

        self.paramset_id = int
        self.valid_params = []
        
    def prep_trade_data(self, side):
        self.get_trade_data()
        self.set_pnl(side)
        self.get_valid_params()
        self.trade_df['DateTime'] = pd.to_datetime(self.trade_df['DateTime'])

    def set_feather_loc(self):
        self.data_loc = (f'{self.data_params.trade_dat_loc}\\{self.data_params.security}\\'
                         f'{self.data_params.time_frame}\\{self.data_params.time_frame}'
                         f'_test_{self.data_params.time_len}')

    def get_trade_data(self):
        print('\nGetting Trade Data')
        self.set_feather_loc()
        self.trade_df = pd.read_feather(f'{self.data_loc}\\{self.data_params.security}_'
                                        f'{self.data_params.time_frame}_Double_Candle_289_trades.feather')
        self.param_df = pd.read_feather(f'{self.data_loc}\\{self.data_params.security}_'
                                        f'{self.data_params.time_frame}_Double_Candle_289_params.feather')
        self.trade_df['DateTime'] = gt.adjust_datetime(self.trade_df['DateTime'])
        self.trade_df = (
            self.trade_df)[self.trade_df['DateTime'].dt.date >=
                           pd.to_datetime(self.data_params.start_train_date).date()]

    def set_pnl(self, side):
        self.trade_df['PnL'] = np.where(self.trade_df['side'] == side,
                                        self.trade_df['entryPrice'] - self.trade_df['exitPrice'],
                                        self.trade_df['exitPrice'] - self.trade_df['entryPrice'])
        self.trade_df['Win_Loss'] = np.where(self.trade_df['PnL'] > 0, 'Win', 'Loss')
        self.trade_df['PnL'] = self.trade_df['PnL']/self.trade_df['entryPrice']*100

    def create_working_df(self, paramset_id=2, side='Bull'):
        print('\nCreating Trades Work Df')
        self.working_df = self.trade_df[(self.trade_df['paramset_id'] == paramset_id) &
                                        (self.trade_df['side'] == side)]
        self.paramset_id = paramset_id

    def subset_test_week(self):
        self.working_df = self.working_df[['DateTime', 'PnL', 'Win_Loss']].reset_index(drop=True)
        self.working_df = self.working_df[(self.working_df['DateTime'].dt.date <= self.curr_test_date.date())]

    def separate_train_test(self, curr_test_date, friday_index):
        print('\nSeparating Train-Test')
        self.curr_test_date = pd.to_datetime(curr_test_date)
        self.get_test_dates()
        self.subset_test_week()

        if friday_index >= 1:
            sample_size = int(len(np.unique(self.working_df['DateTime'].dt.date)) * self.data_params.sample_percent)
            self.train_dates = np.unique(self.working_df['DateTime'].dt.date)
            self.train_dates = random.sample([d for d in self.train_dates], sample_size)
            self.train_dates = self.train_dates + self.get_last_week_train_dates()
        else:
            self.train_dates = [d for d in np.unique(self.working_df['DateTime'].dt.date)]

        self.y_train_df = self.working_df[self.working_df['DateTime'].dt.date.isin(self.train_dates)]
        self.y_test_df = self.working_df[self.working_df['DateTime'].dt.date.isin(self.test_dates)]

    def get_valid_params(self):
        pnl_summary = self.trade_df.groupby(['side', 'paramset_id'], as_index=False)['PnL'].sum()
        pnl_summary = pnl_summary[pnl_summary['PnL'] >=
                                  pnl_summary['PnL'].quantile(self.data_params.min_pnl_percentile)]
        self.valid_params = np.unique(pnl_summary['paramset_id'])

    def get_test_dates(self):
        for i in range(1, 5):
            day_added = (self.curr_test_date - timedelta(days=i)).date()
            self.test_dates.append(day_added)

    def get_last_week_train_dates(self):
        last_week_dates = []
        for i in range(1, 5):
            day_added = (self.curr_test_date - timedelta(days=i+7)).date()
            last_week_dates.append(day_added)

        return last_week_dates

