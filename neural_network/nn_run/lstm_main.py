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
    'other_securities': ['RTY', 'ES', 'YM', 'GC', 'CL'],
    'sides': ['Bull', 'Bear'],
    'time_frame': '15min',
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
    'epochs': 20,
    'batch_size': 32,
    'test_size': 10
}

trade_data_dict = {
    'data_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR'
}


def main():
    lstm_data = ldt.LstmDataHandler(setup_dict, trade_data_dict)
    lstm_data.load_prep_daily_data()
    lstm_data.load_prep_intra_data()
    lstm_data.get_trade_data()
    pnl_summary = lstm_data.trade_data.trade_df.groupby(['side', 'paramset_id'], as_index=False)['PnL'].sum()
    for side in ['Bull', 'Bear']:
        pnl_summary = pnl_summary[pnl_summary['PnL'] > 0]
        for param in np.unique(pnl_summary['paramset_id']):
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

            lstm_model = lmt.LstmOptModel(lstm_model_dict)
            lstm_model.set_intra_len(lstm_data.intra_len)
            lstm_model.build_compile_model(lstm_data)
            lstm_data.win_loss_information()
            model_plot = lstm_model.train_model(lstm_data)
            lstm_model.evaluate_model(lstm_data)
            lstm_model.predict_data(lstm_data, param, side, model_plot)


if __name__ == '__main__':
    main()













