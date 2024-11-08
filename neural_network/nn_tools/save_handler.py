import numpy as np
import pandas as pd
import openpyxl
import os
from datetime import datetime, timedelta
import neural_network.nn_tools.model_tools as modt
from openpyxl.drawing.image import Image
import keras
import pickle


class SaveHandler:
    def __init__(self, mkt_data, lstm_model):
        self.mkt_data = mkt_data
        self.lstm_model = lstm_model
        self.process_handler = None
        self.save_file = None
        self.trade_metrics = None

        self.friday = None
        self.end_date = None
        self.train_date = None
        
        self.model_summary = None
        self.model_plot = None

        self.param_folder = ''
        self.data_folder = ''
        self.model_folder = ''
        self.main_train_path = ''
        self.previous_model_path = None
        self.model_save_path = None

    def check_create_model_folder(self):
        self.param_folder = \
            (f'{self.process_handler.data_params.trade_dat_loc}\\{self.process_handler.data_params.security}\\'
             f'{self.process_handler.data_params.time_frame}\\'
             f'{self.process_handler.data_params.time_frame}_test_{self.process_handler.data_params.time_len}\\'
             f'{self.process_handler.side}')

        self.data_folder = f'{self.param_folder}\\Data'
        self.model_folder = f'{self.param_folder}\\Models'
        self.main_train_path = f'{self.model_folder}\\{self.lstm_model.side}\\param_{self.lstm_model.param}_main_model'

        for folder in [self.param_folder, self.data_folder, self.model_folder]:
            os.makedirs(folder, exist_ok=True)

    def set_model_train_paths(self, friday):
        self.check_create_model_folder()

        previous_friday = pd.to_datetime(friday) - timedelta(days=7)
        previous_friday = previous_friday.strftime(format='%Y-%m-%d')
        self.friday = friday

        self.model_save_path = \
            f'{self.model_folder}\\{self.lstm_model.side}_{self.lstm_model.param}\\param_{self.friday}_model'
        self.previous_model_path = \
            f'{self.model_folder}\\{self.lstm_model.side}_{self.lstm_model.param}\\param_{previous_friday}_model'

        # for folder in [self.model_save_path, self.previous_model_path]:
        #     os.makedirs(folder, exist_ok=True)

    def save_model(self, i):
        print(f'Model Saved: {self.model_save_path}')
        self.lstm_model.model.save(self.model_save_path)
        if i == 0:
            self.lstm_model.model.save(self.main_train_path, save_format='tf')

    def load_prior_friday_model(self):
        last_friday = (pd.to_datetime(self.friday, format='%Y-%m-%d') - timedelta(days=7)).strftime(format='%Y-%m-%d')
        print(f'Loading Prior Week Model: {str(last_friday)}')
        class_weights = self.lstm_model.get_class_weights()
        self.lstm_model.model = keras.models.load_model(
            self.previous_model_path, custom_objects={'loss': modt.weighted_categorical_crossentropy(class_weights),
                                                      'asymmetric_mse': modt.asymmetric_mse})

    def load_current_friday_model(self):
        print(f'Loading Current Week Model: {self.friday}')
        class_weights = self.lstm_model.get_class_weights()
        self.lstm_model.model = (
            keras.models.load_model(self.model_save_path,
                                    custom_objects={'loss': modt.weighted_categorical_crossentropy(class_weights),
                                                    'asymmetric_mse': modt.asymmetric_mse}))

    def save_scalers(self):
        scalers = {
            'y_pnl_scaler': self.mkt_data.y_pnl_scaler,
            'intra_scaler': self.mkt_data.intra_scaler,
        }
        os.makedirs(os.path.dirname(f'{self.model_save_path}\\'), exist_ok=True)
        for key, val in scalers.items():
            with open(f'{self.model_save_path}\\{key}.pkl', 'wb') as f:
                pickle.dump(val, f)
            print(f'Saved {val} Scaler to\n'
                  f'...{self.model_save_path}\\{key}.pkl')

    def load_scalers(self):
        with open(f'{self.previous_model_path}\\y_pnl_scaler.pkl', 'rb') as f:
            self.mkt_data.y_pnl_scaler = pickle.load(f)
        with open(f'{self.previous_model_path}\\intra_scaler.pkl', 'rb') as f:
            self.mkt_data.intra_scaler = pickle.load(f)
        print('Loaded Previous Scalers')

    def save_all_prediction_data(self, side, param, model_dfs, trade_dfs):
        self.save_metrics(side, param, model_dfs, 'Model')
        self.save_metrics(side, param, trade_dfs[0], 'WL')
        self.save_metrics(side, param, trade_dfs[1], 'PnL')
        self.save_plot_to_excel(side)

    def save_metrics(self, side, param, dfs, sheet_name, stack_row=False):
        self.save_file = f'{self.data_folder}\\predictions_{side}_{param}_{self.train_date}.xlsx'
        sheet_name = f'{side}_{sheet_name}'

        if os.path.exists(self.save_file):
            # Load the existing workbook
            book = openpyxl.load_workbook(self.save_file)
            if not book.sheetnames:
                book.create_sheet(sheet_name)
                book.active.title = sheet_name

            with pd.ExcelWriter(self.save_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                start_positions = get_excel_sheet_df_positions(dfs, stack_row)
                write_metrics_to_excel(writer, dfs, sheet_name, start_positions)

        else:
            # Create a new Excel file
            create_new_excel_file(self.save_file, sheet_name)
            with pd.ExcelWriter(self.save_file, engine='openpyxl') as writer:
                start_positions = get_excel_sheet_df_positions(dfs, stack_row)
                write_metrics_to_excel(writer, dfs, sheet_name, start_positions)

    def save_plot_to_excel(self, side):
        file_exists = os.path.exists(self.save_file)
        if not file_exists:
            wb = openpyxl.Workbook()
        else:
            wb = openpyxl.load_workbook(self.save_file)

        # Select a sheet or create a new one
        sheet_name = f'{side}_Model'
        if sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
        else:
            ws = wb.create_sheet(sheet_name)

        img_loc = f'{self.data_folder}\\temp_img.png'
        self.model_plot.fig.savefig(img_loc)
        img = Image(img_loc)

        plot_loc_excel = 'F2'
        if file_exists:
            plot_loc_excel = 'F12'
        ws.add_image(img, plot_loc_excel)

        wb.save(self.save_file)

        if os.path.exists(img_loc):
            os.remove(img_loc)


def write_metrics_to_excel(writer, dfs, sheet_name, start_positions):
    for df, (startrow, startcol) in zip(dfs, start_positions):
        df.to_excel(writer, sheet_name=sheet_name,
                    startrow=startrow, startcol=startcol)


def get_excel_sheet_df_positions(dfs, stack_row):
    start_positions = [(0, 0)]

    if len(dfs) > 1:
        if stack_row:
            start_row = len(dfs[0])
            for df in dfs[1:]:
                start_row += 2
                start_positions.append((start_row, 0))
                start_row += len(df)
        else:
            start_row = 0
            start_col = len(dfs[0].columns) + 2
            for df in dfs[1:]:
                start_positions.append((start_row, start_col))
                start_row += len(df) + 2

    return start_positions


def create_new_excel_file(file_path, sheet_name):
    if not os.path.exists(file_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = sheet_name
        wb.save(file_path)
        wb.close()