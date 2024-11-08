import pandas as pd
import numpy as np
import tensorflow as tf
from data_tools.data_mkt_tools import MktDataHandler
from neural_network.nn_tools.model_tools import LstmOptModel
from data_tools.data_trade_tools import TradeData
from neural_network.nn_tools.save_handler import SaveHandler
from neural_network.nn_tools.process_handler import ProcessHandler, DataParams
from data_tools.data_predictions_tools import ModelOutputdata
from analysis_tools import param_chooser as pc


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

pd.options.mode.chained_assignment = None  # default='warn'


setup_dict = {
    'strategy': 'Double_Candle',
    'model_type': 'LSTM',
    'security': 'NQ',
    'other_securities': ['RTY', 'ES', 'YM'], #, 'GC', 'CL'],
    'sides': ['Bull', 'Bear'],
    'time_frame': '15min',
    'time_length': '20years',
    'data_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data',
    'strat_dat_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double Candles',
    'trade_dat_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR',
    'start_train_date': '2010-04-01',
    'final_test_date': '2024-04-01',
    'start_hour': 6,
    'start_minute': 30,
    'min_pnl_percentile': .3,
    'years_to_train': 3,
    'sample_percent': .33,
    'total_param_sets': 289
}

lstm_model_dict = {
    'epochs': 125,
    'batch_size': 32,
    'test_size': 5,
    'max_accuracy': .95,
    'lstm_i1_nodes': 128,
    'dense_i1_nodes': 96,
    'lstm_i2_nodes': 64,
    'dense_i2_nodes': 64,
    'dense_m1_nodes': 32,
    'adam_optimizer': .00025
}

# 5 min
# lstm_model_dict = {
#     'epochs': 125,
#     'batch_size': 32,
#     'test_size': 5,
#     'max_accuracy': .9,
#     'lstm_i1_nodes': 192,
#     'dense_i1_nodes': 128,
#     'lstm_i2_nodes': 96,
#     'dense_i2_nodes': 64,
#     'dense_m1_nodes': 64,
#     'adam_optimizer': .00025
# }


predict_tf = False

def main():
    data_params = DataParams(setup_dict)
    mkt_data = MktDataHandler(data_params)
    mkt_data.load_prep_data(daily=True)
    mkt_data.load_prep_data(daily=False)
    param_chooser = pc.AlgoParamResults(data_params)
    param_chooser.run_param_chooser()

    for side in ['Bull', 'Bear']:
        trade_data = TradeData(data_params)
        trade_data.prep_trade_data(side)
        mkt_data.set_trade_data(trade_data)

        valid_params = (
            np.array(param_chooser.best_params_df.loc[param_chooser.best_params_df['side'] == side, 'paramset_id']))
        valid_params = np.sort(valid_params)
        print(f'Training {len(valid_params)} Valid Params: \n'
              f'{valid_params}')
        # breakpoint()
        for param in valid_params:
            lstm_model = LstmOptModel(lstm_model_dict, mkt_data, param, side)
            model_output_data = ModelOutputdata(lstm_model, mkt_data, param, side)
            save_handler = SaveHandler(mkt_data, lstm_model)
            process_handler = ProcessHandler(data_params, lstm_model, save_handler, mkt_data, trade_data)
            process_handler.set_lstm_model(lstm_model, side)
            save_handler.process_handler = process_handler

            for i in range(len(process_handler.fridays)):
                friday = process_handler.fridays[i]
                trade_data.create_working_df(paramset_id=param, side=side)
                trade_data.separate_train_test(friday, i)
                save_handler.set_model_train_paths(friday)
                load_scalers = process_handler.decide_load_scalers(i)
                process_handler.prep_training_data(friday, i, load_scalers)
                process_handler.decide_train_predict(param, friday, i)

                if predict_tf:
                    process_handler.load_predict_model(param, side, friday)
                    model_output_data.predict_evaluate_model()
                    model_dfs, trade_dfs = model_output_data.process_prediction_data()
                    save_handler.save_all_prediction_data(side, param, model_dfs, trade_dfs)


if __name__ == '__main__':
    main()













