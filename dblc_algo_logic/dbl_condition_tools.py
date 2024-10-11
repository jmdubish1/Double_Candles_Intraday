import numpy as np
import pandas as pd
import gen_data_tools.general_tools as gt


def dbl_lookback_conds(df_data):
    green_candle = pd.Series(df_data.working_df['Close'] > df_data.working_df['Open'])
    red_candle = ~green_candle

    row_diff = pd.Series(df_data.working_df['Close'] - df_data.working_df['Open'])

    bull_reversal = np.array((row_diff > 0) & (row_diff.shift(1) < 0))
    bull_reversal = gt.fix_tf_arrays(bull_reversal)

    bear_reversal = np.array((row_diff < 0) & (row_diff.shift(1) > 0))
    bear_reversal = gt.fix_tf_arrays(bear_reversal)

    bull_red_good_list = red_candle.rolling(window=df_data.lookback).apply(lambda x: x.all(), raw=True).fillna(True)
    bull_red_good_list = bull_red_good_list.shift(1).fillna(True).astype(int)

    bear_green_good_list = green_candle.rolling(window=df_data.lookback).apply(lambda x: x.all(), raw=True)
    bear_green_good_list = bear_green_good_list.shift(1).fillna(True).astype(int)

    cndl_ratios = np.array(np.abs(row_diff / row_diff.shift(1)).fillna(True))

    row_diff = np.array(row_diff)

    conditions_list = [np.array(df_data.working_df.loc[:, 'DateTime']),
                       bull_reversal, bull_red_good_list,
                       bear_reversal, bear_green_good_list,
                       row_diff, cndl_ratios]

    conditions_cols = ['DateTime',
                       'bull_reversal', 'bull_red_good_list',
                       'bear_reversal', 'bear_green_good_list',
                       'row_diff', 'cndl_ratios']

    conditions_df = pd.DataFrame.from_dict(dict(zip(conditions_cols, conditions_list)))

    return conditions_df


def min_candle_size_conds(df_data):
    cndlsizeseries = (
        pd.Series(np.abs(df_data.working_df['row_diff']) >=
                  (df_data.working_df['Close'] * df_data.mincandlepercent)))

    cndlsizegoodlist = cndlsizeseries.rolling(window=df_data.lookback).apply(lambda x: x.all(), raw=True)
    cndlsizegoodlist = cndlsizegoodlist.shift(1).fillna(True)
    cndlsizegoodlist = gt.fix_tf_arrays(cndlsizegoodlist)

    return cndlsizegoodlist


def fin_candle_size_conds(df_data):
    fincndlsizegoodlist = (
            (pd.Series(np.abs(df_data.working_df['row_diff']) >=
                       (df_data.working_df['Close'] * df_data.finalcandlepercent))) &
            (pd.Series(np.abs(df_data.working_df['row_diff']).shift(1) >=
                       (df_data.working_df['Close'].shift(1) * df_data.finalcandlepercent))))
    fincndlsizegoodlist = gt.fix_tf_arrays(fincndlsizegoodlist)

    return fincndlsizegoodlist


def fin_candle_ratio_conds(df_data):
    fincndlratlist = ((df_data.initial_conds['cndl_ratios'] >= df_data.finalcandleratio) &
                      (df_data.initial_conds['cndl_ratios'] <= 1 / df_data.finalcandleratio))
    fincndlratlist = gt.fix_tf_arrays(fincndlratlist)

    return fincndlratlist


def decide_double_candles(data_df):
    size_ratio_good = (data_df.initial_conds['min_cndl_sz_good'] & data_df.initial_conds['fin_cndl_sz_good'] &
                       data_df.initial_conds['fin_cndl_rat_good'])

    bull_cndl_made = np.array((data_df.initial_conds['bull_reversal'] & data_df.initial_conds['bull_red_good_list'] &
                               size_ratio_good))

    bull_cndl_made[np.isnan(bull_cndl_made)] = 0

    bear_cndl_made = np.array((data_df.initial_conds['bear_reversal'] & data_df.initial_conds['bear_green_good_list'] &
                               size_ratio_good))
    bear_cndl_made[np.isnan(bear_cndl_made)] = 0

    return bull_cndl_made, bear_cndl_made


def create_fastema_exits(data_df):
    close_arr = np.array(data_df.initial_conds['Close'])
    close_shift_arr = np.insert(close_arr[:-1], 0, 0)

    fastema_arr = np.array(data_df.initial_conds['fastEma'])
    fastema_shift_arr = np.insert(fastema_arr[:-1], 0, 0)

    bullexit = (close_arr <= fastema_arr) & (close_shift_arr >= fastema_shift_arr)
    bullexit = gt.fix_tf_arrays(bullexit)
    data_df.initial_conds.loc[:, 'bullExit'] = bullexit

    bearexit = (close_arr >= fastema_arr) & (close_shift_arr <= fastema_shift_arr)
    bearexit = gt.fix_tf_arrays(bearexit)
    data_df.initial_conds.loc[:, 'bearExit'] = bearexit








