import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle


class DataParams:
    def __init__(self, setup_dict):
        self.strategy = setup_dict['strategy']
        self.model_type = setup_dict['model_type']
        self.security = setup_dict['security']
        self.other_securities = setup_dict['other_securities']
        self.sides = setup_dict['sides']
        self.time_frame_test = setup_dict['time_frame_test']
        self.time_frame_train = setup_dict['time_frame_train']
        self.time_len = setup_dict['time_length']
        self.data_loc = setup_dict['data_loc']
        self.strat_dat_loc = setup_dict['strat_dat_loc']
        self.trade_dat_loc = setup_dict['trade_dat_loc']
        self.start_train_date = pd.to_datetime(setup_dict['start_train_date'], format='%Y-%m-%d')
        self.final_test_date = pd.to_datetime(setup_dict['final_test_date'], format='%Y-%m-%d')
        self.start_hour = setup_dict['start_hour']
        self.start_minute = setup_dict['start_minute']
        self.test_period_days = setup_dict['test_period_days']
        self.years_to_train = setup_dict['years_to_train']
        self.sample_percent = setup_dict['sample_percent']
        self.total_param_sets = setup_dict['total_param_sets']


class ProcessHandler:
    def __init__(self, data_params, lstm_model, save_handler, mkt_data, trade_data, retrain_tf):
        self.data_params = data_params
        self.lstm_model = lstm_model
        self.save_handler = save_handler
        self.mkt_data = mkt_data
        self.trade_data = trade_data
        self.test_dates = self.get_test_dates()
        self.train_modeltf = True
        self.retraintf = retrain_tf
        self.predict_datatf = True
        self.prior_traintf = False
        self.load_current_model = False
        self.load_previous_model = False
        self.previous_train_path = None
        self.side = None

    def set_lstm_model(self, lstm_model, side):
        self.lstm_model = lstm_model
        self.side = side

    def get_test_dates(self):
        """Gets a list of all test_date's to train. This should go in another class (possibly processHandler)"""
        end_date = pd.to_datetime(self.data_params.final_test_date, format='%Y-%m-%d')
        end_date = ensure_friday(end_date)
        start_date = end_date - timedelta(weeks=self.data_params.years_to_train*52)

        test_dates = []
        current_date = start_date
        while current_date <= end_date:
            test_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=self.data_params.test_period_days)

        return test_dates

    def adj_test_dates(self, adj_test_date):
        if isinstance(adj_test_date, int):
            self.test_dates = self.test_dates[1:]
        elif isinstance(adj_test_date, pd.Timestamp):
            self.test_dates = [dt for dt in self.test_dates if dt >= adj_test_date]

    def decide_model_to_train(self, param, test_date, use_previous_model):
        current_model_exists = os.path.exists(f'{self.save_handler.model_save_path}\\model.keras')
        previous_model_exists = os.path.exists(f'{self.save_handler.previous_model_path}\\model.keras')
        self.prior_traintf = False
        self.load_current_model = False
        self.load_previous_model = False

        if current_model_exists:
            print(f'Retraining Model: {self.save_handler.model_save_path}')
            self.prior_traintf = True
            self.load_current_model = True
            self.previous_train_path = self.save_handler.model_save_path

            if not self.retraintf:
                print(f'Predicting only: {self.save_handler.previous_model_path}')
                self.train_modeltf = False

        elif previous_model_exists and use_previous_model:
            print(f'Training model from previous model: {self.save_handler.previous_model_path}')
            self.prior_traintf = True
            self.train_modeltf = True
            self.load_previous_model = True
            self.previous_train_path = self.save_handler.previous_model_path

        else:
            print(f'Training New Model...')
            self.train_modeltf = True
            self.prior_traintf = False
        print(f'Training Model: \n...Param: {param} \n...Side: {self.side} \n...Test Date: {test_date}')

    def decide_load_prior_model(self):
        if self.prior_traintf:
            print(f'Loading Prior Model: {self.previous_train_path}')
            if self.load_current_model:
                self.save_handler.load_current_test_date_model()
            elif self.load_previous_model:
                self.save_handler.load_prior_test_date_model()

    def decide_load_scalers(self):
        load_scalers = False
        if self.prior_traintf:
            load_scalers = True
            self.save_handler.load_scalers(self.retraintf)

        else:
            print('Creating New Scalers')

        return load_scalers

    def load_predict_model(self, param, side, test_date):
        if self.predict_datatf:
            print(f'Found data to predict. Predicting trained model: {test_date}')
            if not self.train_modeltf:
                self.save_handler.load_current_test_date_model()
        else:
            print(f'Skipping Model Prediction: \n...Param: {param} \n...Side: {side} \n...Test Date: {test_date}'
                  f'\n***NO TEST TRADES***')

    def set_x_train_test_data(self):
        self.mkt_data.inf_check()
        self.mkt_data.set_x_train_test_datasets()

    def prep_training_data(self, test_date, load_scalers, i):
        self.mkt_data.subset_start_time()
        self.mkt_data.scale_x_data(load_scalers, test_date, self.train_modeltf, i)
        self.mkt_data.scale_y_pnl_data(load_scalers, test_date, self.train_modeltf, i)
        self.mkt_data.onehot_y_wl_data()
        self.save_handler.save_scalers()

    def ph_train_model(self, i):
        if not self.prior_traintf:
            self.lstm_model.build_compile_model()
        else:
            print(f'Loaded Previous Model')
        self.lstm_model.train_model(i, previous_train=self.prior_traintf)
        self.save_handler.save_model(i)

    def modify_op_threshold_temp(self, i, mod_thres=True):
        if i == 0:
            self.lstm_model.opt_threshold = self.lstm_model.lstm_dict['opt_threshold'][self.side]
            self.lstm_model.temperature = self.lstm_model.lstm_dict['temperature'][self.side]
        else:
            opt_df = pd.read_excel(f'{self.save_handler.param_folder}\\best_thresholds.xlsx')

            self.lstm_model.temperature = \
                (opt_df.loc[(opt_df['side'] == self.side) &
                            (opt_df['paramset_id'] == self.lstm_model.param), 'opt_temp'].values)[0]

            if mod_thres:
                self.lstm_model.opt_threshold = \
                    (opt_df.loc[(opt_df['side'] == self.side) &
                                (opt_df['paramset_id'] == self.lstm_model.param), 'opt_threshold'].values)[0]
            else:
                self.lstm_model.opt_threshold = self.lstm_model.lstm_dict['opt_threshold'][self.side]

            # self.lstm_model.opt_threshold = min(round(opt_thres, 3), .575)
            # self.lstm_model.opt_threshold = self.lstm_model.lstm_dict['opt_threshold'][self.side]
            # self.lstm_model.temperature = min(round(opt_temp, 3), 2.0)

    def check_op_thres_temp_params(self):
        temp_predict = False
        opt_file = f'{self.save_handler.param_folder}\\best_thresholds.xlsx'
        if os.path.exists(opt_file):
            opt_df = pd.read_excel(f'{self.save_handler.param_folder}\\best_thresholds.xlsx')
            opt_thres = opt_df.loc[(opt_df['side'] == self.side) &
                                   (opt_df['paramset_id'] == self.lstm_model.param), 'opt_threshold']
            if opt_thres.empty:
                temp_predict = True
        else:
            temp_predict = True

        return temp_predict


def ensure_friday(date):
    weekday = date.weekday()

    if weekday != 4:
        days_until_test_date = (4 - weekday) % 7
        date = date + timedelta(days=days_until_test_date)

    return date















