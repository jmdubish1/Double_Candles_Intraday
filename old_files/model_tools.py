import numpy as np
import pandas as pd
import os
import io
import tensorflow as tf
from datetime import timedelta
from tensorflow.keras.metrics import Precision, Recall, AUC

from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt



class LstmOptModel:
    def __init__(self, lstm_dict, mkt_data, param, side):
        self.lstm_dict = lstm_dict
        self.mkt_data = mkt_data
        self.side = side
        self.param = param

        self.epochs = lstm_dict['epochs']
        self.batch_s = lstm_dict['batch_size']
        self.test_size = lstm_dict['test_size']
        self.max_acc = lstm_dict['max_accuracy']
        self.input_layer_daily = None
        self.input_layer_intraday = None
        self.win_loss_output = None
        self.float_output = None

        self.daily_len = 12
        self.intra_len = 12

        self.model = None
        self.scheduler = None
        self.optimizer = None
        self.input_shapes = None

        self.model_plot = None
        self.model_summary = None

    def build_compile_model(self):
        print('\nBuilding New Model')
        self.build_lstm_model(self.mkt_data.dailydata, self.mkt_data.intradata)
        self.compile_model()

    def get_class_weights(self):
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(self.mkt_data.y_train_wl_df['Win']),
                                             y=self.mkt_data.y_train_wl_df['Win'])
        class_weight_dict_wl = {i: weight for i, weight in enumerate(class_weights)}
        print(f'Class Weights: {class_weight_dict_wl}\n')

        return class_weights

    def compile_model(self, asym_mse=False):
        self.optimizer = Adam(self.lstm_dict['adam_optimizer'])

        class_weights = self.get_class_weights()
        loss_fn_wl = weighted_categorical_crossentropy(class_weights)

        self.model = Model(inputs=[self.input_layer_daily, self.input_layer_intraday],
                           outputs=[self.win_loss_output, self.float_output])

        # if asym_mse:
        #     # auc_metric = AUC(name='auc', curve='PR', num_thresholds=200)
        #     self.model.compile(optimizer=self.optimizer,
        #                        loss={'wl_class': loss_fn_wl,
        #                              'pnl_output': asymmetric_mse},
        #                        metrics={'wl_class': [Precision(thresholds=self.lstm_dict['recall_threshold'],
        #                                                        name='precision'), 'accuracy'],
        #                                 'pnl_output': asymmetric_mse})
        # else:
        #     self.model.compile(optimizer=self.optimizer,
        #                        loss={'wl_class': loss_fn_wl,
        #                              'pnl_output': 'mse'},
        #                        metrics={'wl_class': 'accuracy',
        #                                 'pnl_output': 'mse'})

        if asym_mse:
            # auc_metric = AUC(name='auc', curve='PR', num_thresholds=200)
            self.model.compile(optimizer=self.optimizer,
                               loss={'wl_class': loss_fn_wl,
                                     'pnl': asymmetric_mse},
                               metrics={'wl_class': [bal_accuracy,
                                                     specificity,
                                                     Recall(thresholds=self.lstm_dict['opt_threshold'],
                                                            name='recall')],
                                        'pnl': asymmetric_mse})
        else:
            self.model.compile(optimizer=self.optimizer,
                               loss={'wl_class': loss_fn_wl,
                                     'pnl': 'mse'},
                               metrics={'wl_class': 'accuracy',
                                        'pnl': 'mse'})

        print('New Model Created')
        self.get_model_summary_df()

    def get_input_shapes(self, daily_data, intraday_data):
        """Plus 12 for number of months in year"""
        daily_shape = (self.daily_len, len(daily_data.columns) - 1)
        intraday_shape = (self.mkt_data.intra_len, len(intraday_data.columns) - 1)

        self.input_shapes = [daily_shape, intraday_shape]

    def build_lstm_model(self, daily_data, intraday_data):
        self.get_input_shapes(daily_data, intraday_data)

        # Daily LSTM branch
        self.input_layer_daily = Input(self.input_shapes[0])
        lstm_d1 = LSTM(units=32,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=True,  # Keep this to pass the sequence
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.005),
                       name='lstm_d1')(self.input_layer_daily)

        drop_d1 = Dropout(0.025, name='drop_d1')(lstm_d1)

        # Final LSTM layer for daily data - set return_sequences=False
        lstm_d2 = LSTM(units=24,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.005),
                       name='lstm_d2')(drop_d1)

        dense_d1 = Dense(units=16,
                         activation='tanh',
                         kernel_initializer=GlorotUniform(),
                         kernel_regularizer=l2(0.005),
                         name='dense_d1')(lstm_d2)

        # Intraday LSTM branch
        self.input_layer_intraday = Input(self.input_shapes[1])
        lstm_i1 = LSTM(units=self.lstm_dict['lstm_i1_nodes'],
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=True,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.005),
                       name='lstm_i1')(self.input_layer_intraday)

        drop_i1 = Dropout(0.025, name='drop_i1')(lstm_i1)

        # Final LSTM layer for intraday data - set return_sequences=False
        lstm_i2 = LSTM(units=self.lstm_dict['lstm_i2_nodes'],
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.005),
                       name='lstm_i2')(drop_i1)

        dense_i1 = Dense(units=self.lstm_dict['dense_i1_nodes'],
                         activation='tanh',
                         kernel_initializer=GlorotUniform(),
                         kernel_regularizer=l2(0.005),
                         name='dense_i1')(lstm_i2)

        # Combine the branches
        merged_lstm = Concatenate()([dense_d1, dense_i1])

        drop_m1 = Dropout(0.025, name='drop_m1')(merged_lstm)

        dense_m1 = Dense(units=self.lstm_dict['dense_m1_nodes'],
                         activation='tanh',
                         kernel_initializer=GlorotUniform(),
                         kernel_regularizer=l2(0.005),
                         name='dense_m1')(drop_m1)

        # Output layers
        self.win_loss_output = Dense(2,
                                     activation='sigmoid',
                                     name='wl_class')(dense_m1)

        self.float_output = Dense(units=1,
                                  activation='tanh',
                                  name='pnl')(dense_m1)

    def train_model(self, asym_mse, previous_train):
        if previous_train:
            epochs = 50
            acc_threshold = .99
            self.model.optimizer.learning_rate.assign(self.lstm_dict['adam_optimizer'] / 5)
        else:
            epochs = self.epochs
            acc_threshold = self.max_acc

        lr_scheduler = ReduceLROnPlateau(monitor='loss',
                                         factor=0.8,
                                         patience=2,
                                         min_lr=.0000001,
                                         verbose=2)
        self.model_plot = LivePlotLosses(asym_mse)
        data_gen = CustomDataGenerator(self.mkt_data, self, self.batch_s)
        stop_at_accuracy = StopAtAccuracy(accuracy_threshold=acc_threshold)
        self.model.fit(data_gen,
                       epochs=epochs,
                       verbose=1,
                       callbacks=[lr_scheduler, self.model_plot, stop_at_accuracy])

    def get_model_summary_df(self, printft=True):
        if printft:
            self.model.summary()
            breakpoint()

        summary_buf = io.StringIO()
        self.model.summary(print_fn=lambda x: summary_buf.write(x + "\n"))

        summary_string = summary_buf.getvalue()
        summary_lines = summary_string.split("\n")

        summary_data = []
        for line in summary_lines:
            split_line = list(filter(None, line.split(" ")))
            if len(split_line) > 1:
                summary_data.append(split_line)

        # df_summary = pd.DataFrame(summary_data, columns=['Layer', 'Output Shape', 'Param #', 'Connected To'])
        df_summary = pd.DataFrame(summary_data)
        df_cols = df_summary.iloc[1]
        df_summary = df_summary.iloc[2:].reset_index(drop=True)
        df_summary.columns = df_cols

        self.model_summary = df_summary


"""--------------------------------------------Custom Callbacks Work-------------------------------------------------"""


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, mkt_data, lstm_model, batch_size, train=True):
        self.mkt_data = mkt_data
        self.lstm_model = lstm_model
        self.train_tf = train
        self.sample_ind_list = []
        self.n_samples = 0
        self.batch_size = batch_size
        self.set_attributes()

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        if index == 0:
            np.random.shuffle(self.sample_ind_list)

        start_ind = index * self.batch_size
        end_ind = (index + 1) * self.batch_size
        train_inds = self.sample_ind_list[start_ind:end_ind]
        batch_gen = self.mkt_data.create_batch_input(train_inds,
                                                     self.lstm_model.daily_len,
                                                     self.train_tf)

        x_day_arr, x_intra_arr, y_pnl_arr, y_wl_arr = next(batch_gen)

        return [x_day_arr, x_intra_arr], {'wl_class': y_wl_arr, 'pnl': y_pnl_arr}

    def set_attributes(self):
        if self.train_tf:
            self.sample_ind_list = list(self.mkt_data.y_train_pnl_df.index)
            self.n_samples = len(self.mkt_data.trade_data.y_train_df)

        else:
            self.sample_ind_list = list(self.mkt_data.y_test_pnl_df.index)
            self.n_samples = len(self.mkt_data.trade_data.y_test_df)

    def on_epoch_end(self):
        if self.train_tf:
            np.random.shuffle(self.sample_ind_list)


def asymmetric_mse(y_true, y_pred):
    error = y_true - y_pred
    sign_penalty = tf.where(tf.less(y_true * y_pred, 0), 1.05, 1.0)

    return tf.reduce_mean(sign_penalty * tf.square(error))


def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)

        return tf.reduce_mean(weights * tf.keras.losses.categorical_crossentropy(y_true, y_pred))

    return loss


class LivePlotLosses(Callback):
    def __init__(self, asym_mse=False):
        super(LivePlotLosses, self).__init__()
        self.asym_mse = asym_mse
        self.epochs = []
        self.losses = []
        self.wl_classification_losses = []
        self.pnl_output_losses = []
        self.wl_classification_accuracies = []
        self.pnl_output_mses = []

        plt.ion()  # Interactive mode on for live plotting
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 6))  # Create a 2x2 grid of subplots
        self.fig.tight_layout()

    def on_epoch_end(self, epoch, logs=None):
        if self.asym_mse:
            pnl_mse = 'pnl_output_asymmetric_mse'
        else:
            pnl_mse = 'pnl_output_mse'
        logs = logs or {}
        self.epochs.append(epoch + 1)  # Accumulate epochs
        self.losses.append(logs.get('loss'))
        self.wl_classification_losses.append(logs.get('wl_class_loss'))
        self.pnl_output_losses.append(logs.get(pnl_mse))
        self.wl_classification_accuracies.append(logs.get('wl_class_bal_accuracy'))
        self.pnl_output_mses.append(logs.get(pnl_mse))

        # Clear previous plots
        for ax in self.axs.flat:
            ax.clear()

        # Plot accumulated data for all completed epochs
        self.axs[0, 0].plot(self.epochs, self.losses,
                            label="Total Loss", marker='o')
        self.axs[0, 0].set_title("Total Loss")
        self.axs[0, 0].legend()

        self.axs[0, 1].plot(self.epochs, self.wl_classification_losses,
                            label="WL Classification Loss", marker='o')
        self.axs[0, 1].set_title("WL Classification Loss")
        self.axs[0, 1].legend()

        self.axs[1, 0].plot(self.epochs, self.wl_classification_accuracies,
                            label="WL Classification Accuracy", marker='o')
        self.axs[1, 0].set_title("WL Classification Accuracy")
        self.axs[1, 0].legend()

        self.axs[1, 1].plot(self.epochs, self.pnl_output_mses,
                            label="PnL Output MSE", marker='o')
        self.axs[1, 1].set_title("PnL Output MSE")
        self.axs[1, 1].legend()

        # Draw the updated plots and pause for a short moment to update the plot
        self.fig.canvas.draw()
        plt.pause(0.2)  # Pause briefly to ensure the plot refreshes

    def on_train_end(self, logs=None):
        plt.ioff()  # Turn off interactive mode at the end
        plt.close()
        pass


class StopAtAccuracy(Callback):
    def __init__(self, accuracy_threshold=.99):
        super(StopAtAccuracy, self).__init__()
        self.accuracy_threshold = accuracy_threshold

    def on_epoch_end(self, epoch, logs=None):
        wl_accuracy = logs.get('wl_class_bal_accuracy')
        if wl_accuracy is not None and wl_accuracy >= self.accuracy_threshold:
            print(f"\nReached {self.accuracy_threshold*100}% accuracy, stopping training!")
            self.model.stop_training = True


def bal_accuracy(y_true, y_pred):
    # Convert predictions to binary using a threshold of 0.5 (or adjust as needed)
    y_pred_binary = tf.cast(y_pred > 0.473, tf.float32)

    # Calculate True Positives, True Negatives, False Positives, False Negatives
    true_positives = K.sum(K.cast(y_true * y_pred_binary, tf.float32))
    true_negatives = K.sum(K.cast((1 - y_true) * (1 - y_pred_binary), tf.float32))
    false_positives = K.sum(K.cast((1 - y_true) * y_pred_binary, tf.float32))
    false_negatives = K.sum(K.cast(y_true * (1 - y_pred_binary), tf.float32))

    # Calculate recall for each class
    recall_pos = true_positives / (true_positives + false_negatives + K.epsilon())
    recall_neg = true_negatives / (true_negatives + false_positives + K.epsilon())

    # Compute balanced accuracy
    balanced_acc = (recall_pos + recall_neg) / 2.0
    return balanced_acc


def specificity(y_true, y_pred):
    # Convert predictions to binary using a threshold (default is 0.5)
    y_pred_binary = tf.cast(y_pred > 0.473, tf.float32)

    # Calculate True Negatives and False Positives
    true_negatives = K.sum(K.cast((1 - y_true) * (1 - y_pred_binary), tf.float32))
    false_positives = K.sum(K.cast((1 - y_true) * y_pred_binary, tf.float32))

    # Compute specificity (True Negative Rate)
    specificity_value = true_negatives / (true_negatives + false_positives + K.epsilon())
    return specificity_value


# def wl_combined_loss(y_true, y_pred):
#     loss_fn_weight = 2/5
#     recall_weight = 3/10
#     specificity_weight = 3/10
#     loss = loss_fn_weight *
#     return weight_1 * loss_1(y_true, y_pred) + weight_2 * loss_2(y_true, y_pred)

