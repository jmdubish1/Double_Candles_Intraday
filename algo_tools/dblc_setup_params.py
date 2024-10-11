import pandas as pd
import numpy as np
import os
import warnings
import gen_data_tools.general_tools as gt
import gen_data_tools.ema_tools as et

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


class DblRunParams:
    def __init__(self, param_run_dict):
        self.lookbacks = param_run_dict['lookbacks']
        self.fastEmaLens = param_run_dict['fastEmaLens']
        self.minCndlSizes = self.adj_params(param_run_dict['minCndlSizes'], 10000)
        self.finalCndlSizes = self.adj_params(param_run_dict['finalCndlSizes'], 10000)
        self.finalCndlRatios = self.adj_params(param_run_dict['finalCndlRatios'], 100)
        self.stopLossPercents = self.adj_params(param_run_dict['atrStopLossPercents'], 10)
        self.takeProfitPercents = self.adj_params(param_run_dict['atrTakeProfitPercents'], 10)
        self.daySlowEmaLens = [8]

    def adj_params(self, p_list, adj_):
        return [x/adj_ for x in p_list]


class DblSetupParams:
    def __init__(self, setup_params, param_run_dict):
        self.param_ranges = DblRunParams(param_run_dict)
        self.algo_name = 'Double_Candle'
        self.security = setup_params['security']
        self.timeframe = setup_params['timeframe']
        self.timelength = setup_params['time_length']
        self.begin_date = pd.to_datetime(setup_params['begin_date'], format="%Y/%m/%d")
        self.end_date = pd.to_datetime(setup_params['end_date'], format="%Y/%m/%d")
        self.use_end_date = setup_params['use_end_date']

        self.start_time = pd.to_datetime(setup_params['start_time'], format='%H:%M') #input ex: 07:00
        self.eod_time = pd.to_datetime('15:00', format='%H:%M')
        self.tick_size = setup_params['tick_size']

        self.data_loc = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data'
        self.strat_loc = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR'
        self.file_output = str
        self.intra_day_data_file = str
        self.daily_data_file = str
        self.total_combos = int

    def finish_dblparam_setup(self):
        self.get_tot_combos()
        self.apply_setups()

    def get_tot_combos(self):
        self.total_combos = int(np.prod([len(self.param_ranges.lookbacks),
                                         len(self.param_ranges.fastEmaLens),
                                         len(self.param_ranges.minCndlSizes),
                                         len(self.param_ranges.stopLossPercents),
                                         len(self.param_ranges.daySlowEmaLens)*2,
                                         len(self.param_ranges.takeProfitPercents)+1,
                                         len(self.param_ranges.finalCndlSizes),
                                         len(self.param_ranges.finalCndlRatios)]))

    def apply_setups(self):
        print('Applying Setups')
        self.get_tot_combos()
        self.file_output = f'{self.strat_loc}\\{self.security}\\{self.timeframe}\\{self.timeframe}_test_{self.timelength}'
        os.makedirs(self.file_output, exist_ok=True)
        self.intra_day_data_file = f'{self.data_loc}\\{self.security}_{self.timeframe}_20240505_20040401.txt'
        self.daily_data_file = f'{self.data_loc}\\{self.security}_daily_20240505_20040401.txt'



