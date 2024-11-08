import numpy as np
import pandas as pd
import data_tools.math_tools as mt
from neural_network.nn_tools.model_tools import CustomDataGenerator
from sklearn.metrics import confusion_matrix, classification_report


class ModelOutputdata:
    def __init__(self, lstm_model, mkt_data, param, side):
        self.lstm_model = lstm_model
        self.mkt_data = mkt_data
        self.param = param
        self.side = side
        self.eval_pred_dict = None

    def predict_evaluate_model(self):
        test_generator = CustomDataGenerator(self.mkt_data, self.lstm_model, self.lstm_model.batch_s, train=False)

        # Evaluate the model on the test data
        test_loss, test_wl_loss, test_pnl_loss, wl_accuracy, pnl_output_mse = (
            self.lstm_model.model.evaluate(test_generator))
        wl_predictions, pnl_predictions = self.lstm_model.model.predict(test_generator, verbose=1)

        print(f'Test Loss: {test_loss:.4f}\n'
              f'Win/Loss Classification Loss: {test_wl_loss:.4f}\n'
              f'PNL Output Loss: {test_pnl_loss:.4f}\n'
              f'WL Classification Acc: {wl_accuracy:.4f}\n'
              f'PnL Output MSE: {pnl_output_mse:.4f}')

        self.eval_pred_dict = {
            'test_loss': test_loss,
            'wl_class_loss': test_wl_loss,
            'wl_class_accuracy': wl_accuracy,
            'pnl_output_loss': test_pnl_loss,
            'pnl_output_mse': pnl_output_mse,
            'wl_predictions': wl_predictions,
            'pnl_predictions': pnl_predictions
        }

    def process_prediction_data(self):
        model_metrics = [
            self.eval_pred_dict['test_loss'],
            self.eval_pred_dict['wl_class_loss'],
            self.eval_pred_dict['wl_class_accuracy'],
            self.eval_pred_dict['pnl_output_loss'],
            self.eval_pred_dict['pnl_output_mse']
        ]

        model_metrics = pd.DataFrame(model_metrics).T
        model_metrics.columns = ['test_loss', 'test_wl_loss', 'test_pnl_loss', 'wl_accuracy', 'pnl_output_mse']

        self.prep_predicted_data(self.eval_pred_dict['wl_predictions'], self.eval_pred_dict['pnl_predictions'])

        wl_con_mat = self.mkt_data.get_confusion_matrix_metrics()
        wl_trade_metrics = self.get_trade_metrics()
        wl_dfs = [self.mkt_data.y_test_wl_df, wl_trade_metrics] + [pd.DataFrame(l) for l in wl_con_mat]

        pnl_con_mat = self.mkt_data.get_confusion_matrix_metrics(wl=False)
        pnl_trade_metrics = self.get_trade_metrics(wl=False)
        pnl_dfs = [self.mkt_data.y_test_pnl_df, pnl_trade_metrics] + [pd.DataFrame(l) for l in pnl_con_mat]

        model_dfs = [self.lstm_model.model_summary, model_metrics]
        trade_dfs = [wl_dfs, pnl_dfs]

        return model_dfs, trade_dfs

    def prep_predicted_data(self, wl_predictions, pnl_predictions):
        wl_predictions = self.mkt_data.y_wl_onehot_scaler.inverse_transform(wl_predictions)
        pnl_predictions = self.mkt_data.y_pnl_scaler.inverse_transform(pnl_predictions)
        self.mkt_data.y_test_pnl_df['PnL'] = (
            self.mkt_data.y_pnl_scaler.inverse_transform(self.mkt_data.y_test_pnl_df['PnL'].values.reshape(-1, 1)))

        self.mkt_data.y_test_wl_df['Pred'] = wl_predictions[:, 0]

        self.mkt_data.y_test_wl_df = (
            self.mkt_data.y_test_pnl_df[['DateTime', 'PnL']].merge(self.mkt_data.y_test_wl_df, on='DateTime'))
        self.mkt_data.y_test_pnl_df['Pred'] = pnl_predictions[:, 0]

        self.add_close_to_test_dfs()

        self.mkt_data.y_test_pnl_df = mt.summary_predicted(self.mkt_data.y_test_pnl_df)
        self.mkt_data.y_test_wl_df = mt.summary_predicted(self.mkt_data.y_test_wl_df, wl=True)

        self.prep_actual_wl_cols()
        self.prep_actual_wl_cols(wl=False)

    def add_close_to_test_dfs(self):
        self.mkt_data.y_test_pnl_df = (
            self.mkt_data.y_test_pnl_df.merge(self.mkt_data.security_df[['DateTime', 'Close']], on=['DateTime']))
        self.mkt_data.y_test_wl_df = (
            self.mkt_data.y_test_wl_df.merge(self.mkt_data.security_df[['DateTime', 'Close']], on=['DateTime']))

    def prep_actual_wl_cols(self, wl=True):
        if wl:
            algo_wl = self.mkt_data.y_test_wl_df.apply(lambda row: 'Win' if row['PnL'] > 0 else 'Loss', axis=1)
            self.mkt_data.y_test_wl_df['Loss'] = algo_wl
            self.mkt_data.y_test_wl_df.drop(columns='Win', inplace=True)
            self.mkt_data.y_test_wl_df.rename(columns={'Loss': 'Algo_wl', 'Pred': 'Pred_wl'}, inplace=True)

        else:
            algo_wl = (
                self.mkt_data.y_test_pnl_df.apply(lambda row: 'Win' if row['PnL'] > 0 else 'Loss', axis=1))
            self.mkt_data.y_test_pnl_df['Algo_wl'] = algo_wl

            cols = list(self.mkt_data.y_test_pnl_df.columns)
            cols.insert(2, cols.pop(cols.index('Algo_wl')))
            self.mkt_data.y_test_pnl_df = self.mkt_data.y_test_pnl_df[cols]

            pred_wl = self.mkt_data.y_test_pnl_df.apply(lambda row: 'Win' if row['Pred'] > 0 else 'Loss', axis=1)
            self.mkt_data.y_test_pnl_df['Pred_wl'] = pred_wl

            cols = list(self.mkt_data.y_test_pnl_df.columns)
            cols.insert(4, cols.pop(cols.index('Pred_wl')))
            self.mkt_data.y_test_pnl_df = self.mkt_data.y_test_pnl_df[cols]

    def get_confusion_matrix_metrics(self, wl=True):
        self.win_loss_information()
        if wl:
            df = self.mkt_data.y_test_wl_df
            print('WL Dataset')
        else:
            df = self.mkt_data.y_test_pnl_df
            print('PnL Dataset')

        conf_matrix = confusion_matrix(df['Algo_wl'], df['Pred_wl'], labels=['Win', 'Loss'])
        conf_matrix = pd.DataFrame(conf_matrix, columns=['Pred_Pos', 'Pred_Neg'], index=['Actual_Pos', 'Actual_Neg'])
        conf_matrix = pd.DataFrame(conf_matrix).transpose()
        print('\nConfusion Matrix')
        print(conf_matrix)

        class_report = classification_report(df['Algo_wl'], df['Pred_wl'], target_names=['Win', 'Loss'],
                                             output_dict=True)
        print("\nClassification Report")
        print(class_report)

        return conf_matrix, class_report

    def win_loss_information(self):
        wins = sum(self.mkt_data.y_train_wl_df['Win'])
        losses = sum(self.mkt_data.y_train_wl_df['Loss'])
        wl_ratio = 1 - wins/(wins + losses)
        print(f'\nWin-Loss Ratio to Beat: {wl_ratio:.4f}')

    def get_trade_metrics(self, wl=True):
        self.mkt_data.win_loss_information()
        if wl:
            self.mkt_data.y_test_wl_df.rename({'PnL': 'Algo_PnL'})
            df = self.mkt_data.y_test_wl_df
        else:
            self.mkt_data.y_test_pnl_df.rename({'PnL': 'Algo_PnL'})
            df = self.mkt_data.y_test_pnl_df

        one_dir_algo_stats = one_direction_trade_stats(df, predtf=False)
        one_dir_pred_stats = one_direction_trade_stats(df, predtf=True)
        two_dir_pred_stats = two_direction_trade_stats(df)

        predicted_trade_data = pd.merge(one_dir_algo_stats, one_dir_pred_stats, on='DateTime')
        predicted_trade_data = pd.merge(predicted_trade_data, two_dir_pred_stats, on='DateTime')

        return predicted_trade_data


def one_direction_trade_stats(df, predtf=True):
    if predtf:
        pnl_col = 'Pred'
    else:
        pnl_col = 'Algo'

    correct = (df[f'{pnl_col}_PnL'] > 0).sum()
    tot_trades = (df[f'{pnl_col}_PnL'] != 0).sum()
    percent_correct = correct / tot_trades
    rolling = df[f'{pnl_col}_PnL'].values
    max_pnl = np.max(rolling)
    avg_trade = np.mean(df[f'{pnl_col}_PnL'].values)
    avg_win = np.mean(df.loc[df[f'{pnl_col}_PnL'] > 0, pnl_col])
    avg_loss = np.mean(df.loc[df[f'{pnl_col}_PnL'] < 0, pnl_col])
    max_draw = np.min(df[f'{pnl_col}_MaxDraw'].values)

    stat_dict = {
        'correct': correct,
        'total': tot_trades,
        '%_correct': percent_correct,
        'max_pnl': max_pnl,
        'max_draw': max_draw,
        'avg_trade': avg_trade,
        'avg_win': avg_win,
        'avg_lose': avg_loss
    }

    for key in stat_dict.keys():
        stat_dict[key] = stat_dict[f'one_dir_{key}']
        del stat_dict[key]

    return stat_dict


def two_direction_trade_stats(df):
    correct = (len(df[(df['Pred_wl'] == "Win") & (df['Algo_PnL'] > 0)]) +
               len(df[(df['Pred_wl'] == "Loss") & (df['Algo_PnL'] < 0)]))

    tot_trades = len(df.index)
    percent_correct = correct / tot_trades

    df['Two_Dir_Pred_PnL'] = df.apply(
        lambda row: abs(row['PnL']) if
        (row['Pred_wl'] == 'Win' and row['Algo_PnL'] > 0) or
        (row['Pred_wl'] == 'Loss' and row['Algo_PnL'] < 0) else -abs(row['Algo_PnL']), axis=1)

    rolling = pd.Series(df['Two_Dir_Pred_PnL']).cumsum().values
    df['Two_Dir_Pred_Pnl_Total'] = rolling
    max_pnl = np.max(rolling)
    avg_trade = np.mean(df['Two_Dir_Pred_PnL'].values)
    avg_win = np.mean(df.loc[df['Two_Dir_Pred_PnL'] > 0, 'Two_Dir_Pred_PnL'])
    avg_loss = np.mean(df.loc[df['Two_Dir_Pred_PnL'] < 0, 'Two_Dir_Pred_PnL'])
    df['Two_Dir_Pred_MaxDraw'] = mt.calculate_max_drawdown(df['Two_Dir_Pred_PnL'])
    max_draw = np.min(df['Two_Dir_Pred_MaxDraw'].values)

    stat_dict = {
        'correct': correct,
        'total': tot_trades,
        '%_correct': percent_correct,
        'max_pnl': max_pnl,
        'max_draw': max_draw,
        'avg_trade': avg_trade,
        'avg_win': avg_win,
        'avg_lose': avg_loss
    }

    for key in stat_dict.keys():
        stat_dict[key] = stat_dict[f'two_dir_{key}']
        del stat_dict[key]

    return stat_dict
