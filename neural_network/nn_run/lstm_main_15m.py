import pandas as pd
import numpy as np
import tensorflow as tf
from data_tools.data_mkt_tools2 import MktDataHandler
from neural_network.nn_tools.model_tools2 import LstmOptModel
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
    'time_frame_test': '15min',
    'time_frame_train': '15min',
    'time_length': '20years',
    'data_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data',
    'strat_dat_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double Candles',
    'trade_dat_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR',
    'start_train_date': '2013-04-01',
    'final_test_date': '2024-04-01',
    'start_hour': 6,
    'start_minute': 30,
    'test_period_days': 7*13,
    'years_to_train': 3,
    'sample_percent': 1,
    'total_param_sets': 289
}

lstm_model_dict = {
    'epochs': {'Bull': 80,
               'Bear': 80},
    'batch_size': 8,
    'max_accuracy': .20,
    'lstm_i1_nodes': 160,
    'lstm_i2_nodes': 96,
    'dense_m1_nodes': 64,
    'dense_wl1_nodes': 32,
    'dense_pl1_nodes': 32,
    'adam_optimizer': .0001,
    'prediction_runs': 5,
    'opt_threshold': {'Bull': .28,
                      'Bear': .34},
    'period_lookback': 24,
    'plot_live': False,
    'chosen_params': {'Bull': [],
                      'Bear': []}
}

# 'chosen_params': {'Bull': [44, 45, 76, 87, 89, 108, 120, 129, 157, 169, 171, 213, 104, 188, 252, 234],
#                   'Bear': [165, 157, 160, 74, 121, 118, 166, 250, 124, 226, 100, 220]}


predict_tf = False
retrain_tf = False
use_previous_period_model = False
pnl_output = False
train_bad_paramsets = True


def main():
    data_params = DataParams(setup_dict)
    mkt_data = MktDataHandler(data_params)
    mkt_data.intra_len = lstm_model_dict['period_lookback']
    mkt_data.load_prep_daily_data()
    mkt_data.load_prep_intra_data()
    param_chooser = pc.AlgoParamResults(data_params)
    param_chooser.run_param_chooser()

    for side in setup_dict['sides']:
        trade_data = TradeData(data_params)
        trade_data.prep_trade_data()
        mkt_data.set_trade_data(trade_data)

        good_params = np.array(
            param_chooser.best_params_df.loc[param_chooser.best_params_df['side'] == side, 'paramset_id'])
        other_params = np.array(
            param_chooser.other_params_df.loc[param_chooser.other_params_df['side'] == side, 'paramset_id'])

        if train_bad_paramsets:
            valid_params = np.concatenate((other_params, good_params))
        else:
            valid_params = np.concatenate((good_params, lstm_model_dict['chosen_params'][side]))

        valid_params = sorted(np.unique(valid_params).astype(int))
        param_chooser.set_lstm_nodes(lstm_model_dict)
        param_chooser.save_all_params(valid_params, side)
        print(f'Training {len(valid_params)} Valid Params: \n'
              f'{valid_params}')
        for param in valid_params:
            lstm_model = LstmOptModel(lstm_model_dict, mkt_data, param, side)
            mkt_data.lstm_model = lstm_model
            model_output_data = ModelOutputdata(lstm_model, mkt_data, trade_data, param, side)
            save_handler = SaveHandler(mkt_data, lstm_model, data_params)
            lstm_model.save_handler = save_handler
            process_handler = ProcessHandler(data_params, lstm_model, save_handler, mkt_data, trade_data, retrain_tf)
            process_handler.set_lstm_model(lstm_model, side)
            save_handler.process_handler = process_handler

            print(f'Testing Dates: \n'
                  f'...{process_handler.test_dates}')

            for i in range(len(process_handler.test_dates)):
                test_date = process_handler.test_dates[i]
                trade_data.create_working_df(paramset_id=param, side=side)
                save_handler.set_model_train_paths(test_date)
                trade_data.separate_train_test(test_date)

                process_handler.decide_model_to_train(param, test_date, use_previous_period_model)
                load_scalers = process_handler.decide_load_scalers()
                process_handler.set_x_train_test_data()
                process_handler.prep_training_data(test_date, load_scalers, i)
                process_handler.decide_load_prior_model()

                if process_handler.train_modeltf:
                    param_chooser.adj_lstm_training_nodes(side, param)
                    process_handler.ph_train_model(i)

                if predict_tf and len(trade_data.y_test_df) > 0:
                    model_output_data.predict_evaluate_model(lstm_model_dict['prediction_runs'])
                    model_output_data.agg_prediction_data()
                    model_dfs, trade_dfs = model_output_data.process_prediction_data()

                    save_handler.save_all_prediction_data(side, param, test_date, model_dfs, trade_dfs)
                    trade_data.clear_week_trade_data()
                breakpoint()

if __name__ == '__main__':
    main()













