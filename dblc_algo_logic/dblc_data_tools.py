import numpy as np
import pandas as pd
import os
import warnings
import dblc_algo_logic.dbl_condition_tools as dct
import gen_data_tools.general_tools as gt
import gen_data_tools.ema_tools as et
import algo_tools.stoploss_takeprofit_tools as stt

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


class DblCandlesWorking:
    def __init__(self, setup_params):
        self.setup_params = setup_params
        self.lookback = 0
        self.finalcandleratio = 0.0
        self.fastemalen = 0
        self.mincandlepercent = 0.0
        self.finalcandlepercent = 0.0
        self.stoplosspercent = 0.0
        self.takeprofitpercent = 0.0
        self.dayslow = 0
        self.max_ind = 0
        self.paramset_id = 1
        self.daily_df = None
        self.clean_df = None
        self.working_df = None
        self.intra_ema_df = None
        self.initial_conds = None
        self.fast_ema_df = pd.DataFrame()
        self.stoploss_df = pd.DataFrame()
        self.dayslow_df = None
        self.param_names = ['lookback', 'finalcandleratio', 'fastemalen', 'mincandlepercent', 'finalcandlepercent',
                            'stoplosspercent', 'takeprofitpercent', 'dayslow']

    def apply_data_setups(self):
        self.get_candle_setup_dfs()
        self.get_ema_dfs()
        self.clean_setup_dfs()
        self.build_trade_columns()
        self.finish_working_data()

    def get_candle_setup_dfs(self):
        print('Creating Candle Setup DFs')
        self.clean_df = pd.read_csv(str(self.setup_params.intra_day_data_file))

        self.clean_df['DateTime'] = gt.create_datetime(self.clean_df)
        self.clean_df['Date'] = gt.adjust_dates(self.clean_df['Date'])

        self.daily_df = pd.read_csv(str(self.setup_params.daily_data_file))
        self.daily_df['Date'] = gt.adjust_dates(self.daily_df['Date'])

    def get_ema_dfs(self):
        print('Creating EMAs')
        self.daily_df = et.check_create_emas(self, ema_range=[8], daily=True)
        self.intra_ema_df = et.check_create_emas(self, ema_range=self.setup_params.param_ranges.fastEmaLens)
        self.intra_ema_df['DateTime'] = gt.adjust_datetime(self.intra_ema_df['DateTime'])

    def merge_working_daily(self):
        print('Merging daily and clean_df')
        self.clean_df['Date'] = gt.adjust_dates(self.clean_df['Date'])
        self.daily_df['Date'] = gt.adjust_dates(self.daily_df['Date'])
        self.daily_df = self.daily_df[self.daily_df['Date'].isin(self.clean_df['Date'])]
        self.daily_df.rename(columns={'EMA_8': 'dayEma'}, inplace=True)
        self.clean_df = pd.merge(self.clean_df, self.daily_df[['Date', 'dayEma']], on='Date', how='left')

    def subset_by_dates(self):
        print('Subsetting Dates')
        self.clean_df = self.clean_df.loc[self.clean_df['Date'] >= self.setup_params.begin_date]

        if self.setup_params.use_end_date:
            self.clean_df = self.clean_df.loc[self.clean_df['Date'] < self.setup_params.end_date]

    def clean_setup_dfs(self):
        print('Cleaning DFs')
        cols_to_remove = ["AvgExp12", "AvgExp24", "Bearish_Double_Candle", "Bullish_Double_Candle", "Up", "Down", "Vol",
                          "VolAvg", "OpenInt"]
        self.clean_df = self.clean_df.drop(columns=[col for col in self.clean_df.columns if col in cols_to_remove])

        self.merge_working_daily()
        self.subset_by_dates()

        del self.daily_df

    def build_trade_columns(self):
        print('Building Trade Columns')
        self.clean_df['bullTrade'] = 0
        self.clean_df['bullExit'] = 0

        self.clean_df['bearTrade'] = 0
        self.clean_df['bearExit'] = 0

        self.clean_df['entryPrice'] = 0.0
        self.clean_df['exitPrice'] = 0.0
        self.clean_df['entryInd'] = 0
        self.clean_df['exitInd'] = 0

    def finish_working_data(self):
        """
        :param dbl_setup_params: DblSetupParams class
        """
        self.working_df = self.clean_df.copy(deep=True)
        self.working_df.reset_index(drop=True, inplace=True)
        self.working_df.drop(columns=['Date', 'Time'], inplace=True)
        reorder_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'dayEma', 'bullTrade',
                        'bullExit', 'bearTrade', 'bearExit', 'entryInd', 'entryPrice', 'exitInd', 'exitPrice']
        self.working_df = self.working_df[reorder_cols]

    """---------------------------------------------Trade Work-------------------------------------------------------"""
    def attr_vals(self):
        vals = [self.lookback, self.finalcandleratio, self.fastemalen, self.mincandlepercent, self. finalcandlepercent,
                self.stoplosspercent, self.takeprofitpercent, self.dayslow]
        return vals

    def make_lookback_conds(self):
        self.working_df = self.working_df.merge(dct.dbl_lookback_conds(self),
                                                on=['DateTime'], how='left')

    def apply_eod_exits(self, dbl_working):
        eod_mask = np.array(self.initial_conds['DateTime'].dt.time == dbl_working.eod_time.time())
        self.initial_conds.loc[eod_mask, 'bullExit'] = 1
        self.initial_conds.loc[eod_mask, 'bearExit'] = 1

    """---Trade Work : Min Candle Size ---"""
    def min_candle_setup(self):
        self.working_df['min_cndl_sz_good'] = dct.min_candle_size_conds(self)

    """---Trade Work : Final Candle Size ---"""
    def fin_candle_setup(self):
        self.working_df['fin_cndl_sz_good'] = dct.fin_candle_size_conds(self)
        self.initial_conds = self.working_df.copy(deep=True)

    """---Trade Work : Create Entries---"""
    def fin_candle_ratio_setup(self, dbl_setup):
        self.initial_conds['fin_cndl_rat_good'] = dct.fin_candle_ratio_conds(self)
        self.initial_conds = gt.subset_time(self.initial_conds, dbl_setup, subtract_time=0)

    def create_double_candles(self, dbl_setup):
        self.initial_conds['bullTrade'], self.initial_conds['bearTrade'] = (
            dct.decide_double_candles(self))
        self.remove_eod_entries(dbl_setup)
        self.set_entry_prices()
        self.initial_conds['side'] = gt.get_side(self.initial_conds)

    def remove_eod_entries(self, dbl_setup):
        eod_mask = np.array(self.initial_conds['DateTime'].dt.time == dbl_setup.eod_time.time())
        self.initial_conds.loc[eod_mask, 'bullTrade'] = 0
        self.initial_conds.loc[eod_mask, 'bearTrade'] = 0

    def set_trade_exits(self):
        trade_mask = (self.initial_conds['bullTrade'] == 1) | (self.initial_conds['bearTrade'] == 1)
        self.initial_conds.loc[trade_mask, ['bullExit', 'bearExit']] = 1

    def set_entry_prices(self):
        trade_idx = np.array((self.initial_conds.loc[:, 'bullTrade'] == 1) |
                             (self.initial_conds.loc[:, 'bearTrade'] == 1))
        self.initial_conds.loc[trade_idx, 'entryPrice'] = self.initial_conds.loc[trade_idx, 'Close']

    """---Trade Work : Fast Ema---"""
    def merge_fast_ema(self, dbl_working):
        if 'fastEma' in self.initial_conds.columns:
            self.initial_conds.drop(columns=['fastEma'], inplace=True)
        ema_col_name = f'EMA_{self.fastemalen}'
        self.initial_conds = (
            self.initial_conds.merge(dbl_working.intra_ema_df[['DateTime', ema_col_name]],
                                     on=['DateTime'], how='left'))
        self.initial_conds.rename(columns={ema_col_name: 'fastEma'}, inplace=True)

    def fast_ema_exits(self, dbl_setup):
        dct.create_fastema_exits(self)
        self.apply_eod_exits(dbl_setup)
        self.set_trade_exits()
        self.keep_changes(self.fast_ema_df)
        self.max_ind = max(self.initial_conds.index)

    """---Trade Work : Stoplosses---"""
    def find_stops(self, stop_loss_percent):
        self.takeprofitpercent = 0.0
        self.stoplosspercent = stop_loss_percent
        self.reset_exit_entry(self.fast_ema_df)

        stt.find_stops_bull(self, stop_loss_percent)
        stt.find_stops_bear(self, stop_loss_percent)

        self.keep_changes(self.stoploss_df)

    """---Trade Work : DaySlow---"""
    def apply_dayslow(self, df):
        self.dayslow = 8
        self.dayslow_df = df[['DateTime', 'side', 'dayEma', 'bullTrade', 'bearTrade', 'entryInd',
                              'entryPrice', 'exitPrice', 'exitInd']].copy(deep=True)

        self.dayslow_df.loc[(self.dayslow_df['bearTrade'] == 1) &
                            (self.dayslow_df['dayEma'] <= self.dayslow_df['entryPrice']), 'bearTrade'] = 0

        self.dayslow_df.loc[(self.dayslow_df['bullTrade'] == 1) &
                            (self.dayslow_df['dayEma'] >= self.dayslow_df['entryPrice']), 'bullTrade'] = 0

    """---Trade Work : TakeProfit---"""
    def apply_take_profit(self, take_profit_percent):
        self.reset_exit_entry(self.stoploss_df)
        self.takeprofitpercent = take_profit_percent

        stt.find_tp_bull(self, take_profit_percent)
        stt.find_tp_bear(self, take_profit_percent)

    """---Analyze Work---"""
    def filter_analyze(self, df, result_handler):
        pnl_df = gt.filter_trades(df).values
        self.paramset_id += 1
        paramset_arr = np.repeat([self.paramset_id], len(pnl_df)).reshape(-1, 1)
        pnl_df = np.hstack([paramset_arr, pnl_df])
        result_handler.trade_df = np.vstack([result_handler.trade_df, pnl_df])
        result_handler.paramsets.append([[self.paramset_id] + self.attr_vals()])
        self.dayslow = 0

    """---General---"""
    def reset_exit_entry(self, save_df):
        self.initial_conds['bullTrade'] = save_df['bullTrade'].copy(deep=True)
        self.initial_conds['bearTrade'] = save_df['bearTrade'].copy(deep=True)

        self.initial_conds['bullExit'] = save_df['bullExit'].copy(deep=True)
        self.initial_conds['bearExit'] = save_df['bearExit'].copy(deep=True)

        if 'exitInd' in self.initial_conds.columns:
            self.initial_conds['entryInd'] = save_df['entryInd'].copy(deep=True)
            self.initial_conds['exitInd'] = save_df['exitInd'].copy(deep=True)
            self.initial_conds['exitPrice'] = save_df['exitPrice'].copy(deep=True)

    def keep_changes(self, save_df):
        save_df['bullTrade'] = self.initial_conds['bullTrade'].copy(deep=True)
        save_df['bearTrade'] = self.initial_conds['bearTrade'].copy(deep=True)

        save_df['bullExit'] = self.initial_conds['bullExit'].copy(deep=True)
        save_df['bearExit'] = self.initial_conds['bearExit'].copy(deep=True)

        if 'exitInd' in self.initial_conds.columns:
            save_df['entryInd'] = self.initial_conds['entryInd'].copy(deep=True)
            save_df['exitInd'] = self.initial_conds['exitInd'].copy(deep=True)
            save_df['exitPrice'] = self.initial_conds['exitPrice'].copy(deep=True)


