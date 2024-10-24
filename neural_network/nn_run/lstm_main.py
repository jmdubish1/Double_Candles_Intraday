import pandas as pd
import numpy as np
import tensorflow as tf
import neural_network.nn_tools.lstm_data_tools as ldt
import neural_network.nn_tools.lstm_trade_data_tools as tdt
import neural_network.nn_tools.lstm_model_tools as lmt


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

pd.options.mode.chained_assignment = None  # default='warn'


setup_dict = {
    'model_type': 'LSTM',
    'security': 'NQ',
    'other_securities': ['RTY', 'ES', 'YM'], #, 'GC', 'CL'],
    'sides': ['Bull', 'Bear'],
    'time_frame': '5min',
    'time_length': '20years',
    'data_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data',
    'strat_dat_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double Candles',
    'trade_dat_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR',
    'start_train_date': '2010-04-01',
    'final_train_date': '2024-04-01',
    'start_hour': 6,
    'start_minute': 30
}

lstm_model_dict = {
    'epochs': 125,
    'batch_size': 32,
    'test_size': 5,
    'max_accuracy': .9
}


def main():
    lstm_data = ldt.LstmDataHandler(setup_dict)
    lstm_data.load_prep_daily_data()
    lstm_data.load_prep_intra_data()
    for side in ['Bull', 'Bear']:
        lstm_data.get_trade_data(side)
        pnl_summary = lstm_data.trade_data.trade_df.groupby(['side', 'paramset_id'], as_index=False)['PnL'].sum()
        pnl_summary = pnl_summary[pnl_summary['PnL'] > -2500]
        lstm_data.trade_data.adjust_pnl()
        for param in np.unique(pnl_summary['paramset_id'])[::-1]:
            lstm_model = lmt.LstmOptModel(lstm_model_dict)
            previous_train = lstm_model.check_for_previous_model(lstm_data, side, param)

            if previous_train:
                print(f'Skipping training param: {param} : {side}')
                continue

            lstm_data.trade_data.create_working_df(paramset_id=param, side=side)
            work_df = lstm_data.trade_data.working_df
            if len(work_df) == 0:
                continue
            # lstm_data.intradata.iloc[-1000:].to_excel('intra.xlsx')
            # breakpoint()
            print(f'Working on Param: {param} Side: {side}')
            lstm_data.trade_data.separate_train_test()
            lstm_data.set_x_train_test_datasets()
            lstm_data.scale_x_data()
            lstm_data.scale_y_pnl_data()
            lstm_data.onehot_y_wl_data()

            lstm_model.set_intra_len(lstm_data.intra_len)
            lstm_model.build_compile_model(lstm_data, asym_mse=True)
            lstm_model.train_model(lstm_data, asym_mse=True)
            lstm_model.predict_data_evaluate(lstm_data, param, side)

if __name__ == '__main__':
    main()













