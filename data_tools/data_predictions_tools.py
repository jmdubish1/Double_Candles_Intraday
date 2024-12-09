import numpy as np
import pandas as pd
import data_tools.math_tools2 as mt
from neural_network.nn_tools.model_tools2 import CustomDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option('display.max_columns', None)


class ModelOutputdata:
    def __init__(self, lstm_model, mkt_data, trade_data, param, side):
        self.lstm_model = lstm_model
        self.mkt_data = mkt_data
        self.trade_data = trade_data
        self.param = param
        self.side = side
        self.model_metrics = []
        self.wl_loss = None
        self.wl_nn_binary = None
        self.wl_algo_binary = None
        self.optimal_threshold = None
        self.optimized_temperature = None

    def predict_model(self, runs, thresholdtf=True):
        print(f'Running Model Evaluation...')
        # Evaluate Model
        test_generator = CustomDataGenerator(self.mkt_data, self.lstm_model, self.lstm_model.batch_s, train=False)
        (test_loss, test_wl_loss, test_pnl_loss,
         npv_loss, ppv_loss, auc_loss, huber_loss) = self.lstm_model.model.evaluate(test_generator)

        self.model_metrics = initial_model_metrics()
        self.model_metrics['test_loss'] = test_loss
        self.model_metrics['wl_class_loss'] = test_wl_loss
        self.model_metrics['test_pnl_loss'] = test_pnl_loss
        self.model_metrics['npv_loss'] = npv_loss
        self.model_metrics['ppv_loss'] = ppv_loss
        self.model_metrics['auc_loss'] = auc_loss
        self.model_metrics['huber_loss'] = huber_loss

        # Predict Model
        print(f'Running {runs} Predictions...')

        for i in range(runs):
            if i % 5 == 0:
                print(i)

            test_generator = CustomDataGenerator(self.mkt_data, self.lstm_model, self.lstm_model.batch_s, train=False)
            x_day, x_intra, _, _ = prefab_batches(test_generator)
            x_val = [x_day, x_intra]

            wl_predictions, pnl_predictions = self.lstm_model.model.predict(x_val,
                                                               batch_size=self.lstm_model.batch_s,
                                                               verbose=0)
            wl_predictions = wl_predictions / wl_predictions.sum(axis=1, keepdims=True)

            logit_model = Model(inputs=self.lstm_model.model.input,
                                outputs=self.lstm_model.model.get_layer('logits').output)

            # wl_predictions = self.predict_with_logit_model(logit_model, x_val, self.lstm_model.temperature)

            # pnl_predictions = self.mkt_data.y_pnl_scaler.inverse_transform(pnl_predictions)

            self.model_metrics['wl_predictions'].append(wl_predictions)
            self.model_metrics['pnl_predictions'].append(pnl_predictions)

        if thresholdtf:
            y_pred = np.array(self.model_metrics['wl_predictions'])
            self.get_optimal_threshold(y_pred)

    def predict_optimize_model(self, runs, thresholdtf=True, temp_scaling=True):
        print(f'Running Model Evaluation and Optimization...')
        # Evaluate Model
        test_generator = CustomDataGenerator(self.mkt_data, self.lstm_model, self.lstm_model.batch_s, train=False)
        (test_loss, test_wl_loss, test_pnl_loss,
         npv_loss, ppv_loss, auc_loss, huber_loss) = self.lstm_model.model.evaluate(test_generator)

        self.model_metrics = initial_model_metrics()
        self.model_metrics['test_loss'] = test_loss
        self.model_metrics['wl_class_loss'] = test_wl_loss
        self.model_metrics['test_pnl_loss'] = test_pnl_loss
        self.model_metrics['npv_loss'] = npv_loss
        self.model_metrics['ppv_loss'] = ppv_loss
        self.model_metrics['auc_loss'] = auc_loss
        self.model_metrics['huber_loss'] = huber_loss

        # Predict Model
        print(f'Running {runs} Predictions...')

        for i in range(runs):
            if i % 5 == 0:
                print(i)

            test_generator = CustomDataGenerator(self.mkt_data, self.lstm_model, self.lstm_model.batch_s, train=False)

            if temp_scaling:
                _, pnl_predictions = self.lstm_model.model.predict(test_generator,
                                                                   batch_size=self.lstm_model.batch_s,
                                                                   verbose=0)

                x_day, x_intra, y_wl, _ = prefab_batches(test_generator)
                x_val = [x_day, x_intra]

                logit_model, self.optimized_temperature = (
                    optimize_temperature_scaling(self.lstm_model.model, x_val, y_wl))
                wl_predictions = self.predict_with_logit_model(logit_model, x_val, self.optimized_temperature)

                print(f'Optimized Temperature: {self.optimized_temperature}')
                # breakpoint()

            else:
                wl_predictions, pnl_predictions = self.lstm_model.model.predict(test_generator,
                                                                                batch_size=self.lstm_model.batch_s,
                                                                                verbose=0)

            pnl_predictions = self.mkt_data.y_pnl_scaler.inverse_transform(pnl_predictions)

            self.model_metrics['wl_predictions'].append(wl_predictions)
            self.model_metrics['pnl_predictions'].append(pnl_predictions)

        if thresholdtf:
            y_pred = np.array(self.model_metrics['wl_predictions'])
            self.get_optimal_threshold(y_pred)

    def predict_with_logit_model(self, logit_model, x_val, optimized_temperature):
        logits = logit_model.predict(x_val)
        scaled_logits = logits / optimized_temperature
        wl_predictions = tf.nn.softmax(scaled_logits).numpy()

        return wl_predictions

    def get_optimal_threshold(self, y_pred):
        optimal_thresholds = []
        y_true = np.array(self.mkt_data.y_test_wl_df['Win'])

        for i in range(y_pred.shape[0]):
            y_pred1 = y_pred[i, :, 1].flatten()

            # y_true1 = np.tile(y_true, 1)

            precision, recall, thresholds = precision_recall_curve(y_true, y_pred1)

            f1_scores = 2 * (precision * recall) / (precision + recall + K.epsilon())
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_thresholds.append(optimal_threshold)

        self.optimal_threshold = np.mean(optimal_thresholds)
        optimal_thr_std = np.std(optimal_thresholds)
        self.model_metrics['opt_threshold'] = self.optimal_threshold

        print(f'Period Dates: {self.trade_data.start_period_test_date} - {self.trade_data.curr_test_date}')
        print(f'Current Temperature: {self.lstm_model.temperature}\n'
              f'Current Threshold: {self.lstm_model.opt_threshold}\n'
              f'Optimal Threshold: {self.optimal_threshold}\n'
              f'Op_thres_std: {optimal_thr_std}')

    def opt_thres_stats(self, optimal_thresholds, num_batches=100, batch_size=10):
        means = []
        stds = []
        for i in range(num_batches):
            # Get the current batch
            batch = np.random.choice(optimal_thresholds, batch_size, replace=True)
            if np.isnan(batch).any():
                print(f"Batch {i} contains NaN values.")

            # Calculate mean and std for the batch
            batch_mean = np.mean(np.array(batch))
            batch_std = np.std(np.array(batch))

            # Store the results
            means.append(batch_mean)
            stds.append(batch_std)

        # Convert to arrays (optional)
        means = np.mean(means)
        stds = np.mean(stds)

        print(f'Opt_thres_mean: {means}\n'
              f'Opt_thres_std: {stds}')

        return means, stds

    def agg_prediction_data(self):
        for key, val in self.model_metrics.items():
            if key not in ['wl_predictions', 'pnl_predictions']:
                self.model_metrics[key] = np.mean(self.model_metrics[key])
            elif key == 'wl_predictions':
                data = np.array(self.model_metrics[key])
                data_list1 = []
                data_list2 = []
                for i in range(data.shape[1]):
                    data_list1.append(np.mean(data[:, i, 1]))
                    data_list2.append(np.mean(data[:, i, 0]))

                self.model_metrics[key] = np.array(data_list1).reshape(-1, 1)
                self.wl_loss = np.array(data_list2).reshape(-1, 1)

            else:
                data = np.array(self.model_metrics[key])
                data_list = []
                for i in range(data.shape[1]):
                    data_list.append(np.mean(data[:, i]))
                self.model_metrics[key] = np.array(data_list).reshape(-1, 1)

        for key, val in self.model_metrics.items():
            if key not in ['wl_predictions', 'pnl_predictions']:
                print(f'{key}: {val: 4f}')

    def process_prediction_data(self):
        model_metrics = self.model_metrics.copy()
        key_remove = ['wl_predictions', 'pnl_predictions']
        for k in key_remove:
            model_metrics.pop(k, None)

        model_metrics = pd.DataFrame(model_metrics, index=[0])

        self.prep_predicted_data(self.model_metrics['wl_predictions'],
                                 self.model_metrics['pnl_predictions'])

        wl_con_mat = self.get_confusion_matrix_metrics()
        wl_trade_metrics = self.get_trade_metrics()
        wl_dfs = [self.mkt_data.y_test_wl_df, wl_trade_metrics] + [pd.DataFrame(l) for l in wl_con_mat]

        pnl_con_mat = self.get_confusion_matrix_metrics(wl=False)
        pnl_trade_metrics = self.get_trade_metrics(wl=False)
        pnl_dfs = [self.mkt_data.y_test_pnl_df, pnl_trade_metrics] + [pd.DataFrame(l) for l in pnl_con_mat]

        self.lstm_model.get_model_summary_df(printft=False)
        model_dfs = [self.lstm_model.model_summary, model_metrics]
        trade_dfs = [wl_dfs, pnl_dfs]

        return model_dfs, trade_dfs

    def prep_predicted_data(self, wl_predictions, pnl_predictions):
        self.adjust_algo_pnl_for_close()
        wl_predictions = (wl_predictions >= .5).astype(int)
        # wl_predictions = (wl_predictions >= self.lstm_model.opt_threshold).astype(int)
        #
        # check = np.column_stack((wl_predictions, check))
        # print(check)
        # wl_predictions = check[:, 1]

        self.wl_nn_binary = np.array(wl_predictions).reshape(1, -1)
        self.wl_algo_binary = np.array([1 if i == 1 else 0 for i in self.mkt_data.y_test_wl_df['Win']])

        wl_predictions = ['Win' if pred == 1 else 'Loss' for pred in wl_predictions]

        self.mkt_data.y_test_wl_df.drop(columns='Loss', inplace=True)
        self.mkt_data.y_test_wl_df.rename(columns={'Win': 'Algo_wl'})
        self.mkt_data.y_test_wl_df = (
            self.mkt_data.y_test_wl_df.merge(self.trade_data.analysis_df[['DateTime', 'Algo_PnL']],
                                             on='DateTime', how='left'))

        # self.mkt_data.y_test_wl_df['Pred'] = wl_predictions.flatten()
        self.mkt_data.y_test_wl_df['Pred'] = wl_predictions

        self.mkt_data.y_test_pnl_df = (
            self.mkt_data.y_test_pnl_df.merge(self.trade_data.analysis_df[['DateTime', 'Algo_PnL']],
                                              on='DateTime', how='left'))

        self.mkt_data.y_test_pnl_df['Pred'] = pnl_predictions.flatten()

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
            # print(self.mkt_data.y_test_wl_df)
            algo_wl = self.mkt_data.y_test_wl_df.apply(lambda row: 'Win' if row['Algo_PnL'] > 0 else 'Loss', axis=1)
            self.mkt_data.y_test_wl_df['Loss'] = algo_wl
            self.mkt_data.y_test_wl_df.drop(columns='Win', inplace=True)
            self.mkt_data.y_test_wl_df.rename(columns={'Loss': 'Algo_wl', 'Pred': 'Pred_wl'}, inplace=True)

        else:
            algo_wl = (
                self.mkt_data.y_test_pnl_df.apply(lambda row: 'Win' if row['Algo_PnL'] > 0 else 'Loss', axis=1))
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

        try:
            class_report = classification_report(df['Algo_wl'], df['Pred_wl'], target_names=['Win', 'Loss'],
                                                 output_dict=True)
        except ValueError:
            w_count = sum([1 for i in df['Pred_wl'] if i == 'Win'])
            if (w_count == 0) or (w_count == len(df['Pred_wl'])):
                class_report = {'Win': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 4.0},
                                'Loss': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 4.0}
                                }
            else:
                class_report = {'Win': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 4.0},
                                'Loss':
                                    {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 4.0},
                                }

        print("\nClassification Report")
        print(class_report)

        return conf_matrix, class_report

    def win_loss_information(self):
        wins = sum(self.mkt_data.y_train_wl_df['Win'])
        losses = sum(self.mkt_data.y_train_wl_df['Loss'])
        wl_ratio = 1 - wins/(wins + losses)
        print(f'\nLoss Ratio to Beat: {wl_ratio:.4f}')

        y_true = self.wl_algo_binary
        y_pred = self.wl_nn_binary

        true_positives = np.sum(y_true * y_pred)
        true_negatives = np.sum((1 - y_true) * (1 - y_pred))
        false_positives = np.sum((1 - y_true) * y_pred)
        false_negatives = np.sum(y_true * (1 - y_pred))

        print(f'True Pos: {true_positives}')
        print(f'False Pos: {false_positives}')
        print(f'True Neg: {true_negatives}')
        print(f'False Neg: {false_negatives}')
        print(f'Loss Ratio LSTM: {true_negatives / len(y_true)}')

    def get_trade_metrics(self, wl=True):
        if wl:
            df = self.mkt_data.y_test_wl_df
        else:
            df = self.mkt_data.y_test_pnl_df
        # df.rename(columns={'PnL': 'Algo_PnL'}, inplace=True)

        one_dir_algo_stats = one_direction_trade_stats(df, predtf=False)
        one_dir_pred_stats = one_direction_trade_stats(df, predtf=True)
        two_dir_pred_stats = two_direction_trade_stats(df)

        predicted_trade_data = pd.concat([one_dir_algo_stats, one_dir_pred_stats, two_dir_pred_stats])

        return predicted_trade_data

    def adjust_algo_pnl_for_close(self):
        self.trade_data.analysis_df = (
            self.trade_data.analysis_df.merge(self.mkt_data.security_df[['DateTime', 'Close']]))
        self.trade_data.analysis_df['Algo_PnL'] = (
                self.trade_data.analysis_df['Algo_PnL'] * self.trade_data.analysis_df['Close'])/100


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
    avg_win = np.mean(df.loc[df[f'{pnl_col}_PnL'] > 0, f'{pnl_col}_PnL'])
    avg_loss = np.mean(df.loc[df[f'{pnl_col}_PnL'] < 0, f'{pnl_col}_PnL'])
    max_draw = np.min(df[f'{pnl_col}_MaxDraw'].values)
    expect_val = percent_correct * avg_win + (1 - percent_correct) * avg_loss

    stat_dict = {
        'correct': correct,
        'total': tot_trades,
        '%_correct': percent_correct,
        'max_pnl': max_pnl,
        'max_draw': max_draw,
        'avg_trade': avg_trade,
        'avg_win': avg_win,
        'avg_lose': avg_loss,
        'expect_val': expect_val
    }

    stat_dict = pd.DataFrame(stat_dict, index=[pnl_col])

    return stat_dict


def two_direction_trade_stats(df):
    correct = (len(df[(df['Pred_wl'] == "Win") & (df['Algo_PnL'] > 0)]) +
               len(df[(df['Pred_wl'] == "Loss") & (df['Algo_PnL'] < 0)]))

    tot_trades = len(df.index)
    percent_correct = correct / tot_trades

    df['Two_Dir_Pred_PnL'] = df.apply(
        lambda row: abs(row['Algo_PnL']) if
        (row['Pred_wl'] == 'Win' and row['Algo_PnL'] > 0) or
        (row['Pred_wl'] == 'Loss' and row['Algo_PnL'] < 0) else -abs(row['Algo_PnL']), axis=1)

    rolling = pd.Series(df['Two_Dir_Pred_PnL']).cumsum().values
    df['Two_Dir_Pred_PnL_Total'] = pd.Series(df['Two_Dir_Pred_PnL']).cumsum().values
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

    stat_dict = pd.DataFrame(stat_dict, index=['Two_dir'])

    return stat_dict


def initial_model_metrics():
    metrics_dict = {
        'test_loss': 0.0,
        'wl_class_loss': 0.0,
        'wl_class_accuracy': 0.0,
        'wl_class_specificity': 0.0,
        'pnl_output_loss': 0.0,
        'pnl_output_mse': 0.0,
        'opt_threshold': 0.0,
        'wl_predictions': [],
        'pnl_predictions': []
    }

    return metrics_dict


def softmax_pred(arr):
    return arr / arr.sum(axis=1, keepdims=True)


def optimize_temperature_scaling(model, x_val, y_val, act_type='sigmoid'):
    # Extract logits
    logit_model = Model(inputs=model.input, outputs=model.get_layer('logits').output)
    logits = logit_model.predict(x_val)

    # Ensure y_val is one-hot encoded if logits have multiple classes
    if len(y_val.shape) == 1 or y_val.shape[-1] != logits.shape[-1]:
        y_val = tf.one_hot(tf.cast(y_val, tf.int32), depth=logits.shape[-1])

    # Initialize temperature as a trainable variable
    temperature = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Define the loss function (negative log-likelihood)
    def loss_fn(act_type):
        scaled_logits = logits / temperature
        loss = None
        if act_type == 'softmax':
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_val, logits=scaled_logits)
        elif act_type == 'sigmoid':
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.argmax(y_val, axis=-1), logits=scaled_logits)
        return tf.reduce_mean(loss)

    # Optimize the temperature
    for _ in range(100):  # Number of steps
        optimizer.minimize(loss_fn, [temperature])

    return logit_model, temperature.numpy()


def prefab_batches(test_generator):
    X_day = []
    X_intra = []
    Y_wl = []
    Y_pnl = []

    for x_batch, y_batch in test_generator:
        x_day_arr, x_intra_arr = x_batch
        X_day.append(x_day_arr)
        X_intra.append(x_intra_arr)
        Y_wl.append(y_batch['wl_class'])
        Y_pnl.append(y_batch['pnl'])

    # Combine all batches into single arrays
    X_day = np.concatenate(X_day, axis=0)
    X_intra = np.concatenate(X_intra, axis=0)
    Y_wl = np.concatenate(Y_wl, axis=0)
    Y_pnl = np.concatenate(Y_pnl, axis=0)

    return X_day, X_intra, Y_wl, Y_pnl