import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import openpyxl
import neural_network.nn_tools.model_tools as mt


class DataParams:
    def __init__(self, setup_dict):
        self.strategy = setup_dict['strategy']
        self.model_type = setup_dict['model_type']
        self.security = setup_dict['security']
        self.other_securities = setup_dict['other_securities']
        self.sides = setup_dict['sides']
        self.time_frame = setup_dict['time_frame']
        self.time_len = setup_dict['time_length']
        self.data_loc = setup_dict['data_loc']
        self.strat_dat_loc = setup_dict['strat_dat_loc']
        self.trade_dat_loc = setup_dict['trade_dat_loc']
        self.start_train_date = pd.to_datetime(setup_dict['start_train_date'], format='%Y-%m-%d')
        self.final_test_date = pd.to_datetime(setup_dict['final_test_date'], format='%Y-%m-%d')
        self.start_hour = setup_dict['start_hour']
        self.start_minute = setup_dict['start_minute']
        self.min_pnl_percentile = setup_dict['min_pnl_percentile']
        self.years_to_train = setup_dict['years_to_train']
        self.sample_percent = setup_dict['sample_percent']
        self.total_param_sets = setup_dict['total_param_sets']


class ProcessHandler:
    def __init__(self, data_params, lstm_model, save_handler, mkt_data, trade_data):
        self.data_params = data_params
        self.lstm_model = lstm_model
        self.save_handler = save_handler
        self.mkt_data = mkt_data
        self.trade_data = trade_data
        self.fridays = self.get_fridays()
        self.train_modeltf = True
        self.predict_datatf = True
        self.previous_traintf = False
        self.previous_train_path = None
        self.side = None

    def set_lstm_model(self, lstm_model, side):
        self.lstm_model = lstm_model
        self.side = side

    def get_fridays(self):
        """Gets a list of all Friday's to train. This should go in another class (possibly processHandler)"""
        end_date = pd.to_datetime(self.data_params.final_test_date, format='%Y-%m-%d')
        end_date = ensure_friday(end_date)
        start_date = end_date - timedelta(weeks=self.data_params.years_to_train*52)

        fridays = []
        current_date = start_date
        while current_date <= end_date:
            fridays.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(weeks=1)

        return fridays

    def adj_test_dates(self, adj_test_date):
        if isinstance(adj_test_date, int):
            self.fridays = self.fridays[1:]
        elif isinstance(adj_test_date, pd.Timestamp):
            self.fridays = [dt for dt in self.fridays if dt >= adj_test_date]

    def check_previous_train(self):
        if os.path.exists(self.save_handler.model_save_path):
            self.train_modeltf = False
        else:
            self.train_modeltf = True

        if self.train_modeltf:
            if os.path.exists(self.save_handler.previous_model_path):
                print(f'Training model from previous model: {self.save_handler.previous_model_path}')
                self.previous_traintf = True
                self.previous_train_path = self.save_handler.previous_model_path
                self.save_handler.load_prior_friday_model()

            elif os.path.exists(self.save_handler.main_train_path):
                print(f'Training model from base model: {self.save_handler.main_train_path}')
                self.previous_traintf = True
                self.previous_train_path = self.save_handler.main_train_path
            else:
                print(self.save_handler.model_folder)
                print(f'Training new model: \n...{self.save_handler.model_save_path}')

    def decide_train_predict(self, param, friday, i):
        self.check_previous_train()
        if len(self.trade_data.working_df) == 0:
            self.predict_datatf = False

        if self.train_modeltf:
            self.save_handler.save_scalers()
            if self.previous_traintf:
                print(f'Found Previous Week Model...')
                self.save_handler.load_prior_friday_model()
            else:
                print(f'Training New Model...')
            print(f'Training Model: \n...Param: {param} \n...Side: {self.side} \n...Test Date: {friday}')

            self._train_model(i)

    def load_predict_model(self, param, side, friday):
        if self.predict_datatf:
            print(f'Found data to predict. Predicting trained model: {friday}')
            if not self.train_modeltf:
                self.save_handler.load_current_friday_model()
        else:
            print(f'Skipping Model Prediction: \n...Param: {param} \n...Side: {side} \n...Test Date: {friday}'
                  f'\n***NO TEST TRADES***')

    def prep_training_data(self, friday, i, load_scalers):
        self.trade_data.separate_train_test(friday, i)
        self.mkt_data.set_x_train_test_datasets()
        self.mkt_data.scale_x_data(load_scalers, friday)
        self.mkt_data.scale_y_pnl_data(load_scalers, friday)
        self.mkt_data.onehot_y_wl_data()

    def _train_model(self, i):
        self.lstm_model.build_compile_model(asym_mse=True)
        self.lstm_model.train_model(asym_mse=True, previous_train=self.previous_traintf)
        self.save_handler.save_model(i)

    def decide_load_scalers(self, i):
        load_scalers = False
        if i != 0:
            load_scalers = True
            self.save_handler.load_scalers()

        return load_scalers


def ensure_friday(date):
    weekday = date.weekday()

    if weekday != 4:
        days_until_friday = (4 - weekday) % 7
        date = date + timedelta(days=days_until_friday)

    return date
















