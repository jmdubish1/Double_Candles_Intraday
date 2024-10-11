import numpy as np


def standard_exit(exit_prices, exit_inds, close_prices, start_idx, idx_curr_exit):
    exit_prices[start_idx] = close_prices[idx_curr_exit]
    exit_inds[start_idx] = idx_curr_exit


def find_stops_bull(data_df, stop_loss_percent):
    bull_trade_idx = np.array(data_df.initial_conds.loc[data_df.initial_conds['bullTrade'] == 1].index.to_list() +
                              [data_df.max_ind])
    idx_dummy = np.array(range(0, data_df.max_ind+1))

    curr_exits = np.array(data_df.initial_conds['bullExit'])
    sl_prices = (np.array(data_df.initial_conds['entryPrice']) -
                 (data_df.initial_conds['ATR']) * stop_loss_percent)

    close_prices = np.array(data_df.initial_conds['Close'])
    exit_prices = np.array(data_df.initial_conds['exitPrice'])
    exit_inds = np.array(data_df.initial_conds['exitInd'])

    low_prices = np.array(data_df.initial_conds['Low'])

    if len(bull_trade_idx) > 0:
        for k in range(1, len(bull_trade_idx)):
            start_idx = bull_trade_idx[k-1]
            end_idx = bull_trade_idx[k]+1

            all_exit_slice = curr_exits[start_idx+1:end_idx]
            if len(all_exit_slice) == 0:
                data_df.initial_conds.at[start_idx, 'bullTrade'] = 0
                continue

            idx_dummy_slice = idx_dummy[start_idx+1:end_idx]
            sl_price = sl_prices[start_idx]
            sl_exit_slice = np.array(low_prices[start_idx+1:end_idx] <= sl_price).astype(int)

            next_curr_exit = np.argmax(all_exit_slice)
            idx_curr_exit = idx_dummy_slice[next_curr_exit]

            if sum(sl_exit_slice) > 0:
                next_sl_exit = np.argmax(sl_exit_slice)
                idx_sl_exit = idx_dummy_slice[next_sl_exit]

                if idx_sl_exit <= idx_curr_exit:
                    exit_prices[start_idx] = sl_price
                    exit_inds[start_idx] = idx_sl_exit
                else:
                    standard_exit(exit_prices, exit_inds, close_prices, start_idx, idx_curr_exit)
            else:
                standard_exit(exit_prices, exit_inds, close_prices, start_idx, idx_curr_exit)

    data_df.initial_conds.loc[:, 'exitPrice'] = exit_prices
    data_df.initial_conds.loc[:, 'exitInd'] = exit_inds
    data_df.initial_conds.loc[:, 'entryInd'] = data_df.initial_conds.index.to_list()


def find_stops_bear(data_df, stop_loss_percent):
    bear_trade_idx = (np.array(data_df.initial_conds.loc[data_df.initial_conds['bearTrade'] == 1].index.to_list() +
                               [data_df.max_ind]))
    idx_dummy = np.array(range(0, data_df.max_ind+1))

    curr_exits = np.array(data_df.initial_conds['bearExit'])
    sl_prices = (np.array(data_df.initial_conds['entryPrice']) +
                 (data_df.initial_conds['ATR']) * stop_loss_percent)

    close_prices = np.array(data_df.initial_conds['Close'])
    exit_prices = np.array(data_df.initial_conds['exitPrice'])
    exit_inds = np.array(data_df.initial_conds['exitInd'])

    high_prices = np.array(data_df.initial_conds['High'])

    if len(bear_trade_idx) > 0:
        for k in range(1, len(bear_trade_idx)):
            start_idx = bear_trade_idx[k-1]
            end_idx = bear_trade_idx[k]+1

            all_exit_slice = curr_exits[start_idx+1:end_idx]
            idx_dummy_slice = idx_dummy[start_idx+1:end_idx]

            if len(all_exit_slice) == 0:
                data_df.initial_conds.at[idx_dummy_slice[start_idx], 'bearTrade'] = 0
                continue

            sl_price = sl_prices[start_idx]
            sl_exit_slice = np.array(high_prices[start_idx+1:end_idx] >= sl_price).astype(int)

            next_curr_exit = np.argmax(all_exit_slice)
            idx_curr_exit = idx_dummy_slice[next_curr_exit]

            if sum(sl_exit_slice) > 0:
                next_sl_exit = np.argmax(sl_exit_slice)
                idx_sl_exit = idx_dummy_slice[next_sl_exit]

                if idx_sl_exit <= idx_curr_exit:
                    exit_prices[start_idx] = sl_price
                    exit_inds[start_idx] = idx_sl_exit
                else:
                    standard_exit(exit_prices, exit_inds, close_prices, start_idx, idx_curr_exit)
            else:
                standard_exit(exit_prices, exit_inds, close_prices, start_idx, idx_curr_exit)

    data_df.initial_conds.loc[:, 'exitPrice'] = exit_prices
    data_df.initial_conds.loc[:, 'exitInd'] = exit_inds
    data_df.initial_conds.loc[:, 'entryInd'] = data_df.initial_conds.index.to_list()


def find_tp_bull(data_df, take_profit_percent):
    bull_trade_idx = (np.array(data_df.initial_conds.loc[data_df.initial_conds['bullTrade'] == 1].index.to_list() +
                               [data_df.max_ind]))
    idx_dummy = np.array(range(0, data_df.max_ind+1))
    tp_prices = (np.array(data_df.initial_conds['entryPrice']) +
                 (data_df.initial_conds['ATR']) * take_profit_percent)
    exit_prices = np.array(data_df.initial_conds['exitPrice'])
    exit_inds = np.array(data_df.initial_conds['exitInd'])

    high_prices = np.array(data_df.initial_conds['High'])

    if len(bull_trade_idx) > 0:
        for k in range(1, len(bull_trade_idx)):
            start_idx = bull_trade_idx[k-1]
            end_idx = bull_trade_idx[k]+1

            tp_price = tp_prices[start_idx]
            tp_exit_slice = np.array(high_prices[start_idx+1:end_idx] >= tp_price).astype(int)

            next_curr_exit = exit_inds[start_idx]
            idx_dummy_slice = idx_dummy[start_idx+1:end_idx]

            if sum(tp_exit_slice) > 0:
                next_tp_exit = np.argmax(tp_exit_slice)
                idx_tp_exit = idx_dummy_slice[next_tp_exit]

                if idx_tp_exit < next_curr_exit:
                    exit_prices[start_idx] = tp_price
                    exit_inds[start_idx] = idx_tp_exit

    data_df.initial_conds.loc[:, 'exitPrice'] = exit_prices
    data_df.initial_conds.loc[:, 'exitInd'] = exit_inds
    data_df.initial_conds.loc[:, 'entryInd'] = data_df.initial_conds.index.to_list()


def find_tp_bear(data_df, take_profit_percent):
    bear_trade_idx = (np.array(data_df.initial_conds.loc[data_df.initial_conds['bearTrade'] == 1].index.to_list() +
                               [data_df.max_ind]))
    idx_dummy = np.array(range(0, data_df.max_ind+1))
    tp_prices = (np.array(data_df.initial_conds['entryPrice']) -
                 (data_df.initial_conds['ATR']) * take_profit_percent)
    exit_prices = np.array(data_df.initial_conds['exitPrice'])
    exit_inds = np.array(data_df.initial_conds['exitInd'])

    low_prices = np.array(data_df.initial_conds['Low'])

    if len(bear_trade_idx) > 0:
        for k in range(1, len(bear_trade_idx)):
            start_idx = bear_trade_idx[k-1]
            end_idx = bear_trade_idx[k]+1

            tp_price = tp_prices[start_idx]
            tp_exit_slice = np.array(low_prices[start_idx+1:end_idx] <= tp_price).astype(int)

            next_curr_exit = exit_inds[start_idx]

            idx_dummy_slice = idx_dummy[start_idx+1:end_idx]

            if sum(tp_exit_slice) > 0:
                next_tp_exit = np.argmax(tp_exit_slice)
                idx_tp_exit = idx_dummy_slice[next_tp_exit]
                """This could be refined to reflect if the exit was an SL or not. If it is an SL, then simply
                <, if it is a close, then <=. Either way, using just < will create a more restrictive and less
                risky output"""
                if idx_tp_exit < next_curr_exit:
                    exit_prices[start_idx] = tp_price
                    exit_inds[start_idx] = idx_tp_exit

    data_df.initial_conds.loc[:, 'exitPrice'] = exit_prices
    data_df.initial_conds.loc[:, 'exitInd'] = exit_inds
    data_df.initial_conds.loc[:, 'entryInd'] = data_df.initial_conds.index.to_list()


