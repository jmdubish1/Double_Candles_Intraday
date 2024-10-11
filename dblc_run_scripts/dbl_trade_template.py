import dblc_algo_logic.dblc_make_trades as dmt

setup_params = {
    'security': 'NQ',
    'timeframe': '15min',
    'time_length': '20years',
    'begin_date': '2004/04/01',
    'end_date': '2024/05/01',
    'use_end_date': True,
    'start_time': '08:00',
    'tick_size': .25
}

param_run_dict = {
    'lookbacks': [3],
    'fastEmaLens': range(10, 21, 2),
    'minCndlSizes': [1],
    'finalCndlSizes': [1],
    'finalCndlRatios': [30],
    'atrStopLossPercents': range(5, 21, 5),
    'atrTakeProfitPercents': range(10, 36, 5)
}



combo_start = 1

def main():
    dmt.make_trades(setup_params, param_run_dict, combo_start)

if __name__ == '__main__':
    main()
