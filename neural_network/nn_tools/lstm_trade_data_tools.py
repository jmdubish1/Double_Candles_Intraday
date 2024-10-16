import numpy as np
import random
import pandas as pd
import gen_data_tools.general_tools as gt

class TradeData:
    def __init__(self, trade_dict):
        self.trade_dict = trade_dict
        self.data_loc = str
        self.trade_df = None
        self.param_df = None

        self.working_df = None
        self.train_dates = []
        self.y_train_df = None
        self.test_dates = []
        self.y_test_df = None

        self.paramset_id = int

    def set_feather_loc(self):
        self.data_loc = (f'{self.trade_dict["trade_dat_loc"]}\\{self.trade_dict["security"]}\\'
                         f'{self.trade_dict["time_frame"]}\\{self.trade_dict["time_frame"]}'
                         f'_test_{self.trade_dict["time_length"]}')

    def get_trade_data(self):
        print('\nGetting Trade Data')
        self.set_feather_loc()
        self.trade_df = pd.read_feather(f'{self.data_loc}\\{self.trade_dict["security"]}_'
                                        f'{self.trade_dict["time_frame"]}_Double_Candle_289_trades.feather')
        self.param_df = pd.read_feather(f'{self.data_loc}\\{self.trade_dict["security"]}_'
                                        f'{self.trade_dict["time_frame"]}_Double_Candle_289_params.feather')
        self.trade_df['DateTime'] = gt.adjust_datetime(self.trade_df['DateTime'])
        self.trade_df = (
            self.trade_df)[self.trade_df['DateTime'].dt.date >=
                           pd.to_datetime(self.trade_dict['start_train_date']).date()]

    def set_pnl(self, side):
        self.trade_df['PnL'] = np.where(self.trade_df['side'] == side,
                                        self.trade_df['entryPrice'] - self.trade_df['exitPrice'],
                                        self.trade_df['exitPrice'] - self.trade_df['entryPrice'])
        self.trade_df['Win_Loss'] = np.where(self.trade_df['PnL'] > 0, 'Win', 'Loss')

    def adjust_pnl(self):
        self.trade_df['PnL'] = self.trade_df['PnL']/self.trade_df['entryPrice']*100

    def create_working_df(self, paramset_id=2, side='Bull'):
        print('\nCreating Trades Work Df')
        self.working_df = self.trade_df[(self.trade_df['paramset_id'] == paramset_id) &
                                        (self.trade_df['side'] == side)]
        self.working_df = self.working_df[['DateTime', 'PnL', 'Win_Loss']].reset_index(drop=True)
        self.paramset_id = paramset_id

    def separate_train_test(self, test_percent=.2):
        print('\nSeparating Train-Test')
        uniq_dates_full = [d for d in np.unique(self.working_df['DateTime'].dt.date)]
        last_2month_test = uniq_dates_full[-92:]
        uniq_tf_split = uniq_dates_full[:-92]

        # test_size = int(len(uniq_tf_split)*test_percent)
        # self.test_dates = random.sample(uniq_tf_split, test_size)
        self.test_dates = self.test_dates + last_2month_test
        self.train_dates = [item for item in uniq_tf_split if item not in self.test_dates]

        self.y_train_df = self.working_df[self.working_df['DateTime'].dt.date.isin(self.train_dates)]
        self.y_test_df = self.working_df[self.working_df['DateTime'].dt.date.isin(self.test_dates)]


