import random
import time
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2, l1
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import ReduceLROnPlateau


class LstmOptModel:
    def __init__(self, lstm_dict):
        self.epochs = lstm_dict['epochs']
        self.batch_s = lstm_dict['batch_size']
        self.test_size = lstm_dict['test_size']
        self.input_layer_daily = None
        self.input_layer_intraday = None
        self.win_loss_output = None
        self.float_output = None

        self.daily_len = 23
        self.intra_len = 30

        self.model = None
        self.scheduler = None
        self.optimizer = None
        self.input_shapes = None

    def set_scheduler(self):
        self.scheduler = CustLRSchedule(initial_learning_rate=.03, decay_rate=0.00001)

    def set_intra_len(self, intra_len):
        self.intra_len = intra_len

    def build_compile_model(self, daily_data, intraday_data):
        # print(full_data.prev_model_save)
        # if not full_data.prev_model_loaded:
        print('Building New Model')
        self.set_scheduler()
        self.build_lstm_model(daily_data, intraday_data)
        self.compile_model()

    def compile_model(self):
        self.optimizer = Adam(.01)
        # self.model = Model(inputs=self.input_layer,
        #                    outputs=self.float_output)
        self.model = Model(inputs=[self.input_layer_daily, self.input_layer_intraday],
                           outputs=[self.win_loss_output, self.float_output])
        self.model.compile(optimizer=self.optimizer,
                           loss={'wl_classification': 'categorical_crossentropy',
                                 'pnl_output': 'mse'},
                           metrics={'wl_classification': 'accuracy',
                                    'pnl_output': 'mse'})
        print('New Model Created')
        self.model.summary()

    def get_input_shapes(self, daily_data, intraday_data):
        """Plus 12 for number of months in year"""
        daily_shape = (self.daily_len, len(daily_data.columns) - 1)
        intraday_shape = (self.intra_len, len(intraday_data.columns) - 1)

        self.input_shapes = [daily_shape, intraday_shape]

    def build_lstm_model(self, daily_data, intraday_data):
        self.get_input_shapes(daily_data, intraday_data)

        # Daily LSTM branch
        self.input_layer_daily = Input(self.input_shapes[0])
        lstm_d1 = LSTM(units=64,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=True,  # Keep this to pass the sequence
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.025),
                       name='lstm_d1')(self.input_layer_daily)

        batch_n1 = BatchNormalization(name='batch_n1')(lstm_d1)

        drop_d1 = Dropout(0.1, name='drop_d1')(batch_n1)

        # Final LSTM layer for daily data - set return_sequences=False
        lstm_d2 = LSTM(units=32,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.01),
                       name='lstm_d2')(drop_d1)

        dense_d1 = Dense(units=32,
                         activation='tanh',
                         kernel_initializer=GlorotUniform(),
                         name='dense_d1')(lstm_d2)

        # Intraday LSTM branch
        self.input_layer_intraday = Input(self.input_shapes[1])
        lstm_i1 = LSTM(units=128,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=True,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.01),
                       name='lstm_i1')(self.input_layer_intraday)

        batch_n2 = BatchNormalization(name='batch_n2')(lstm_i1)

        drop_i1 = Dropout(0.1, name='drop_i1')(batch_n2)

        # Final LSTM layer for intraday data - set return_sequences=False
        lstm_i2 = LSTM(units=64,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.01),
                       name='lstm_i2')(drop_i1)

        dense_i1 = Dense(units=64,
                         activation='tanh',
                         kernel_initializer=GlorotUniform(),
                         name='dense_i1')(lstm_i2)

        # Combine the branches
        merged_lstm = Concatenate()([dense_d1, dense_i1])

        dense_m1 = Dense(units=32,
                         activation='tanh',
                         kernel_initializer=GlorotUniform(),
                         name='dense_m1')(merged_lstm)

        # Output layers
        self.win_loss_output = Dense(2,
                                     activation='tanh',
                                     name='wl_classification')(dense_m1)

        self.float_output = Dense(units=1,
                                  activation='tanh',
                                  name='pnl_output')(dense_m1)

    def train_model(self, lstm_data):
        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.75, patience=1, min_lr=1e-6, verbose=1)
        data_gen = CustomDataGenerator(lstm_data, self, self.batch_s)
        self.model.fit(data_gen,
                       epochs=self.epochs,
                       verbose=1,
                       callbacks=[lr_scheduler])

    def evaluate_model(self, lstm_data):
        test_generator = CustomDataGenerator(lstm_data, self, self.batch_s, train=False)

        # Evaluate the model on the test data
        test_loss, test_wl_loss, test_pnl_loss, wl_accuracy, pnl_output_mse = self.model.evaluate(test_generator)

        print(f'Test Loss: {test_loss:.4f}')
        print(f'Win/Loss Classification Loss: {test_wl_loss:.4f}')
        print(f'PNL Output Loss: {test_pnl_loss:.4f}')
        print(f'WL Classification Acc: {wl_accuracy:.4f}')
        print(f'PnL Output MSE: {pnl_output_mse:.4f}')

    def predict_data(self, lstm_data):
        test_generator = CustomDataGenerator(lstm_data, self, self.batch_s, train=False)
        predictions = self.model.predict(test_generator, verbose=1)

        wl_predictions, pnl_predictions = predictions

        wl_predictions = lstm_data.y_wl_onehot_scaler.inverse_transform(wl_predictions)
        pnl_predictions = lstm_data.y_pnl_scaler.inverse_transform(pnl_predictions)

        print("\nWin/Loss Predictions shape:", wl_predictions.shape)
        print("PnL Predictions shape:", pnl_predictions.shape)

        lstm_data.y_test_wl_df['Pred'] = wl_predictions[:, 0]
        lstm_data.y_test_pnl_df['Pred'] = pnl_predictions[:, 0]

        print(f'\nFirst 10 Win/Loss predictions:, {lstm_data.y_test_wl_df.tail(15)}')
        print(f'\nFirst 10 PnL predictions:, {lstm_data.y_test_pnl_df.tail(15)}')


"""--------------------------------------------Custom Callbacks Work-------------------------------------------------"""


class CustLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def __call__(self, step):
        # Define the custom learning rate schedule here.
        # For example, a simple exponential decay:
        return self.initial_learning_rate / (1 + self.decay_rate * step)

    def get_config(self):
        # Required if you need to save the schedule.
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_rate': self.decay_rate
        }


"""--------------------------------------------Custom Callbacks Work-------------------------------------------------"""


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, lstm_data, lstm_model, batch_size, train=True):
        self.lstm_data = lstm_data
        self.lstm_model = lstm_model
        self.train_test = train
        self.sample_ind_list = []
        self.n_samples = 0
        self.batch_size = batch_size
        self.set_attributes()

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        start_ind = index * self.batch_size
        end_ind = (index + 1) * self.batch_size
        train_inds = self.sample_ind_list[start_ind:end_ind]
        batch_gen = self.lstm_data.create_batch_input(train_inds,
                                                      self.lstm_model.daily_len,
                                                      self.lstm_model.intra_len,
                                                      self.train_test)

        x_day_arr, x_intra_arr, y_pnl_arr, y_wl_arr = next(batch_gen)

        return [x_day_arr, x_intra_arr], {'wl_classification': y_wl_arr, 'pnl_output': y_pnl_arr}

    def set_attributes(self):
        if self.train_test:
            self.sample_ind_list = list(self.lstm_data.y_train_pnl_df.index)
            self.n_samples = len(self.lstm_data.trade_data.y_train_df)
            np.random.shuffle(self.sample_ind_list)
        else:
            self.sample_ind_list = list(self.lstm_data.y_test_pnl_df.index)
            self.n_samples = len(self.lstm_data.trade_data.y_test_df)

    def on_epoch_end(self):
        # Optionally shuffle the data at the end of each epoch
        if self.train_test:
            np.random.shuffle(self.sample_ind_list)

