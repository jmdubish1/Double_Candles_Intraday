import algo_tools.dblc_setup_params as dsp
import dblc_algo_logic.dblc_data_tools as ddt
import gen_data_tools.general_tools as gt
import time


def make_trades(setup_params, param_run_dict, combo_start):
    dbl_params = dsp.DblSetupParams(setup_params, param_run_dict)
    dbl_params.finish_dblparam_setup()

    dbl_working = ddt.DblCandlesWorking(dbl_params)
    dbl_working.apply_data_setups()
    result_handler = gt.AggHandler(dbl_params, combo_start)

    for lookback in dbl_params.param_ranges.lookbacks:
        dbl_working.lookback = lookback
        dbl_working.make_lookback_conds()
        dbl_working.working_df['ATR'] = gt.create_atr(dbl_working.working_df, 8)
        dbl_working.working_df = gt.subset_time(dbl_working.working_df,
                                                dbl_params,
                                                subtract_time=1)

        start_time = time.time()
        for mincandlepercent in dbl_params.param_ranges.minCndlSizes:
            print(f'Min Candle Size: {mincandlepercent}')
            dbl_working.min_candle_setup()

            for finalcandlepercent in dbl_params.param_ranges.finalCndlSizes:
                print(f'Final Candle Size: {finalcandlepercent}')
                dbl_working.finalcandlepercent = finalcandlepercent
                dbl_working.fin_candle_setup()
                dbl_working.working_df = gt.subset_time(dbl_working.working_df, dbl_params)

                for finalcandleratio in dbl_params.param_ranges.finalCndlRatios:
                    print(f'Starting Main Logic: Final Candle Ratio: {finalcandleratio}')
                    dbl_working.finalcandleratio = finalcandleratio
                    dbl_working.fin_candle_ratio_setup(dbl_params)
                    dbl_working.create_double_candles(dbl_params)

                    result_handler.decide_save(5000)
                    for fast_ema_len in dbl_params.param_ranges.fastEmaLens:
                        print(f'FastEma Len: {fast_ema_len}')
                        dbl_working.fastemalen = fast_ema_len
                        dbl_working.merge_fast_ema(dbl_working)
                        dbl_working.fast_ema_exits(dbl_params)

                        for stop_loss_percent in dbl_params.param_ranges.stopLossPercents:
                            dbl_working.find_stops(stop_loss_percent)
                            dbl_working.filter_analyze(dbl_working.initial_conds, result_handler)

                            dbl_working.apply_dayslow(dbl_working.initial_conds)
                            dbl_working.filter_analyze(dbl_working.dayslow_df, result_handler)

                            result_handler.combo += 2

                            for take_profit_percent in dbl_params.param_ranges.takeProfitPercents:
                                result_handler.combo += 2
                                if take_profit_percent < stop_loss_percent:
                                    continue
                                else:
                                    dbl_working.takeprofitpercent = take_profit_percent
                                    dbl_working.apply_take_profit(take_profit_percent)
                                    dbl_working.filter_analyze(dbl_working.initial_conds, result_handler)

                                    dbl_working.apply_dayslow(dbl_working.initial_conds)
                                    dbl_working.filter_analyze(dbl_working.dayslow_df, result_handler)

                        print(f"{result_handler.combo}/{result_handler.total_combos}")
                        print("Seconds TP: %f" % (time.time() - start_time))

    result_handler.decide_save(0)


