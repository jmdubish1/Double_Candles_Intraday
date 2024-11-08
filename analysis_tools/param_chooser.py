import itertools

import numpy as np
import pandas as pd


class AlgoParamResults:
    def __init__(self, data_params):
        self.end_date = pd.to_datetime(data_params.final_test_date, format='%Y-%m-%d')
        self.trade_folder = (f'{data_params.trade_dat_loc}\\{data_params.security}\\{data_params.time_frame}\\'
                             f'{data_params.time_frame}_test_{data_params.time_len}')
        self.trades_file = (f'{self.trade_folder}\\{data_params.security}_{data_params.time_frame}_'
                            f'{data_params.strategy}_{data_params.total_param_sets}_trades.feather')
        self.trade_df = None
        self.param_file = (f'{self.trade_folder}\\{data_params.security}_{data_params.time_frame}_'
                           f'{data_params.strategy}_{data_params.total_param_sets}_params.feather')
        self.param_df = None
        self.pnl_df = None
        self.pnl_df_save_file = f'{self.trade_folder}\\{data_params.security}_{data_params.time_frame}_pnl.xlsx'
        self.best_params_df = []
        self.best_params_save_file = (f'{self.trade_folder}\\'
                                      f'{data_params.security}_{data_params.time_frame}_best_params.xlsx')

    def load_files(self):
        self.trade_df = pd.read_feather(self.trades_file)
        self.param_df = pd.read_feather(self.param_file)

    def set_pnl(self):
        self.pnl_df = self.trade_df.copy()
        self.pnl_df['PnL'] = np.where(self.trade_df['side'] == 'Bear',
                                      self.trade_df['entryPrice'] - self.trade_df['exitPrice'],
                                      self.trade_df['exitPrice'] - self.trade_df['entryPrice'])

    def subset_date_agg_pnl(self):
        self.pnl_df = self.pnl_df[self.pnl_df['DateTime'] < self.end_date]
        self.pnl_df = self.pnl_df.groupby(['side', 'paramset_id'], as_index=False)['PnL'].sum()

    def merge_pnl_params(self):
        self.pnl_df = pd.merge(self.pnl_df, self.param_df, on='paramset_id')

    def get_best_params(self):

        best_params = []
        for side in ['Bear', 'Bull']:
            temp_side = self.pnl_df[self.pnl_df['side'] == side]

            # Best PnLs for EMAs
            temp_side = temp_side.sort_values(['fastEmaLen', 'PnL'], ascending=[True, False])
            temp_ema = temp_side.groupby('fastEmaLen').head(2)
            best_params.append(temp_ema)
            temp_day_ema = temp_side.groupby('dayEma').head(4)
            best_params.append(temp_day_ema)

        best_params = pd.concat(best_params)
        best_params.drop_duplicates(inplace=True)

        self.best_params_df = best_params

    def save_dfs(self):
        self.pnl_df.to_excel(self.pnl_df_save_file)
        self.best_params_df.to_excel(self.best_params_save_file)

    def run_param_chooser(self):
        self.load_files()
        self.set_pnl()
        self.subset_date_agg_pnl()
        self.merge_pnl_params()
        self.get_best_params()
        self.save_dfs()

def group_pnls_for_params(df, param_list):
    standard_set = ['side', 'atrStopLossPercents', 'atrTakeProfitPercents']
    if len(param_list) > 0:
        [standard_set.append(i) for i in param_list]

    df = df.groupby(standard_set)['PnL'].mean()
    df = pd.DataFrame(df).reset_index()
    df.sort_values(by=['PnL'], ascending=False, inplace=True)

    return df


def get_best_pnls(df):
    best_pnls = []
    for _, row in df.iterrows():
        if row['PnL'] > 0:
            best_pnls.append(row)
        else:
            while len(best_pnls) < 2:
                best_pnls.append(row)
    return best_pnls
