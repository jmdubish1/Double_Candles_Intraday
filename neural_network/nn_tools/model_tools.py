import numpy as np
import pandas as pd
import os
import io
import tensorflow as tf
from datetime import timedelta
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import data_tools.data_mkt_tools as lmt
from openpyxl.drawing.image import Image
import openpyxl


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

        self.daily_len = 23
        self.intra_len = 24

        self.model = None
        self.scheduler = None
        self.optimizer = None
        self.input_shapes = None

        self.model_plot = None

    def build_compile_model(self, asym_mse=False):
        print('\nBuilding New Model')
        self.build_lstm_model(self.mkt_data.dailydata, self.mkt_data.intradata)
        self.compile_model(asym_mse)

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

        if asym_mse:
            self.model.compile(optimizer=self.optimizer,
                               loss={'wl_classification': loss_fn_wl,
                                     'pnl_output': asymmetric_mse},
                               metrics={'wl_classification': 'accuracy',
                                        'pnl_output': asymmetric_mse})
        else:
            self.model.compile(optimizer=self.optimizer,
                               loss={'wl_classification': loss_fn_wl,
                                     'pnl_output': 'mse'},
                               metrics={'wl_classification': 'accuracy',
                                        'pnl_output': 'mse'})

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
        lstm_d1 = LSTM(units=48,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=True,  # Keep this to pass the sequence
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.005),
                       name='lstm_d1')(self.input_layer_daily)

        batch_n1 = BatchNormalization(name='batch_n1')(lstm_d1)

        drop_d1 = Dropout(0.05, name='drop_d1')(batch_n1)

        # Final LSTM layer for daily data - set return_sequences=False
        lstm_d2 = LSTM(units=32,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.005),
                       name='lstm_d2')(drop_d1)

        dense_d1 = Dense(units=32,
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

        batch_n2 = BatchNormalization(name='batch_n2')(lstm_i1)

        drop_i1 = Dropout(0.05, name='drop_i1')(batch_n2)

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

        drop_m1 = Dropout(0.05, name='drop_m1')(merged_lstm)

        dense_m1 = Dense(units=self.lstm_dict['dense_m1_nodes'],
                         activation='tanh',
                         kernel_initializer=GlorotUniform(),
                         kernel_regularizer=l2(0.005),
                         name='dense_m1')(drop_m1)

        # Output layers
        self.win_loss_output = Dense(2,
                                     activation='softmax',
                                     name='wl_classification')(dense_m1)

        self.float_output = Dense(units=1,
                                  activation='tanh',
                                  name='pnl_output')(dense_m1)

    def train_model(self, asym_mse, previous_train):
        if previous_train:
            epochs = 50
            acc_threshold = .975
        else:
            epochs = self.epochs
            acc_threshold = self.max_acc

        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.75, patience=2, min_lr=.00001, verbose=1)
        self.model_plot = LivePlotLosses(asym_mse)
        data_gen = CustomDataGenerator(self.mkt_data, self, self.batch_s)
        stop_at_accuracy = StopAtAccuracy(accuracy_threshold=acc_threshold)
        self.model.fit(data_gen,
                       epochs=epochs,
                       verbose=1,
                       callbacks=[lr_scheduler, self.model_plot, stop_at_accuracy])

    def get_model_summary_df(self):
        self.model.summary()

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

        return [x_day_arr, x_intra_arr], {'wl_classification': y_wl_arr, 'pnl_output': y_pnl_arr}

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
    sign_penalty = tf.where(tf.less(y_true * y_pred, 0), 1.25, 1.0)

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
        self.wl_classification_losses.append(logs.get('wl_classification_loss'))
        self.pnl_output_losses.append(logs.get(pnl_mse))
        self.wl_classification_accuracies.append(logs.get('wl_classification_accuracy'))
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


class StopAtAccuracy(Callback):
    def __init__(self, accuracy_threshold=.95):
        super(StopAtAccuracy, self).__init__()
        self.accuracy_threshold = accuracy_threshold

    def on_epoch_end(self, epoch, logs=None):
        wl_accuracy = logs.get('wl_classification_accuracy')
        if wl_accuracy is not None and wl_accuracy >= self.accuracy_threshold:
            print(f"\nReached {self.accuracy_threshold*100}% accuracy, stopping training!")
            self.model.stop_training = True
