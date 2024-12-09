import neural_network.nn_tools.loss_functions as lf
import numpy as np
import pandas as pd
import io
from tensorflow.keras.metrics import Precision, Recall, AUC
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Concatenate, Reshape, GRU)
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class LstmOptModel:
    def __init__(self, lstm_dict, mkt_data, param, side):
        self.lstm_dict = lstm_dict
        self.mkt_data = mkt_data
        self.side = side
        self.param = param
        self.temperature = lstm_dict['temperature'][side]

        self.epochs = self.lstm_dict['epochs'][self.side]
        self.batch_s = lstm_dict['batch_size']
        self.max_acc = lstm_dict['max_accuracy']
        self.input_layer_daily = None
        self.input_layer_intraday = None
        self.win_loss_output = None
        self.float_output = None

        self.daily_len = lstm_dict['period_lookback']
        self.intra_len = lstm_dict['period_lookback']

        self.model = None
        self.scheduler = None
        self.optimizer = None
        self.input_shapes = None

        self.model_plot = None
        self.model_summary = None
        self.save_handler = None
        self.opt_threshold = self.lstm_dict['opt_threshold'][self.side]

    def build_compile_model(self):
        print(f'\nBuilding New Model\n'
              f'Param ID: {self.param}')
        self.build_lstm_model()
        self.compile_model()
        self.model.summary()

    def get_class_weights(self):
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(self.mkt_data.y_train_wl_df['Win']),
                                             y=self.mkt_data.y_train_wl_df['Win'])

        class_weight_dict_wl = {i: weight for i, weight in enumerate(class_weights)}
        print(f'Class Weights: {class_weight_dict_wl}\n')
        # self.opt_threshold = class_weights[0]/class_weights[1]
        return class_weights

    def get_input_shapes(self):
        """Plus 12 for number of months in year"""
        daily_shape = (self.daily_len, len(self.mkt_data.dailydata.columns) - 1)
        # intraday_shape = (self.mkt_data.intra_len, len(intraday_data.columns) - 1)
        intraday_shape = (self.intra_len, len(self.mkt_data.intradata.columns) - 1)

        self.input_shapes = [daily_shape, intraday_shape]

    def compile_model(self):
        self.optimizer = Adam(self.lstm_dict['adam_optimizer'], clipnorm=1.0)
        threshold = self.opt_threshold
        class_weights = self.get_class_weights()
        combined_wl_loss = lf.comb_focal_wce_f1(beta=1.5,
                                                opt_threshold=threshold,
                                                class_weights=class_weights)
        # wce_accuracy = lf.weighted_categorical_crossentropy(class_weights)
        # huber_loss = lf.weighted_huber_loss()
        # loss_fl = lf.focal_loss()
        npv_fn = lf.negative_predictive_value(threshold)
        auc = lf.weighted_auc(class_weights)
        ppv_fn = lf.positive_predictive_value(threshold)

        self.model = Model(inputs=[self.input_layer_daily, self.input_layer_intraday],
                           outputs=[self.win_loss_output, self.float_output])

        self.model.compile(optimizer=self.optimizer,
                           loss={'wl_class': combined_wl_loss,
                                 'pnl': 'mse'},
                           metrics={'wl_class': [npv_fn,
                                                 ppv_fn,
                                                 auc],
                                    'pnl': 'mse'},
                           loss_weights={'wl_class': 0.60,
                                         'pnl': 0.40})

        print('New Model Created')
        self.get_model_summary_df()

    def build_lstm_model(self):
        self.get_input_shapes()

        # Daily LSTM branch
        self.input_layer_daily = Input(self.input_shapes[0])

        lstm_d1 = LSTM(units=24,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,  # Keep this to pass the sequence
                       kernel_initializer=GlorotUniform(),
                       # kernel_regularizer=l2(0.01),
                       kernel_regularizer=l2(0.0001),
                       name='lstm_d1')(self.input_layer_daily)

        # drop_d1 = Dropout(0.10, name='drop_d1')(lstm_d1)
        #
        # lstm_d2 = LSTM(units=32,
        #                activation='tanh',
        #                recurrent_activation='sigmoid',
        #                return_sequences=False,
        #                kernel_initializer=GlorotUniform(),
        #                kernel_regularizer=l2(0.01),
        #                # kernel_regularizer=l2(0.01),
        #                name='lstm_d2')(drop_d1)

        self.input_layer_intraday = Input(self.input_shapes[1])

        lstm_i1 = LSTM(units=self.lstm_dict['lstm_i1_nodes'],
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(0.0001),
                       # kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                       name='lstm_i1')(self.input_layer_intraday)

        # drop_i1 = Dropout(0.10, name='drop_i1')(lstm_i1)
        #
        # lstm_i2 = LSTM(units=self.lstm_dict['lstm_i2_nodes'],
        #                activation='tanh',
        #                recurrent_activation='sigmoid',
        #                return_sequences=False,
        #                kernel_initializer=GlorotUniform(),
        #                kernel_regularizer=l2(0.01),
        #                # kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
        #                name='lstm_i2')(drop_i1)

        merged_lstm = Concatenate(axis=-1,
                                  name='concatenate_timesteps')([lstm_d1, lstm_i1],)

        dense_m1 = Dense(units=self.lstm_dict['dense_m1_nodes'],
                         activation='tanh',
                         kernel_initializer=GlorotUniform(),
                         kernel_regularizer=l2(0.001),
                         # kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                         name='dense_m1')(merged_lstm)

        drop_i1 = Dropout(0.10, name='drop_m1')(dense_m1)

        dense_wl1 = Dense(units=self.lstm_dict['dense_wl1_nodes'],
                          activation='sigmoid',
                          kernel_initializer=GlorotUniform(),
                          kernel_regularizer=l2(0.01),
                          # kernel_regularizer=l1(0.01),
                          name='dense_wl1')(drop_i1)

        dense_pl1 = Dense(units=self.lstm_dict['dense_pl1_nodes'],
                          activation='tanh',
                          kernel_initializer=GlorotUniform(),
                          # kernel_regularizer=l2(0.01),
                          kernel_regularizer=l2(0.01),
                          name='dense_pl1')(drop_i1)

        logit_layer = Dense(2,
                            activation=None,
                            name='logits')(dense_wl1)
        #
        temp_scale_wl = TemperatureScalingLayer(self.temperature,
                                                name='temp_scaling')(logit_layer)

        # Output layers
        self.win_loss_output = Dense(2,
                                     activation='sigmoid',
                                     name='wl_class')(temp_scale_wl)

        self.float_output = Dense(units=1,
                                  activation='tanh',
                                  name='pnl')(dense_pl1)

    def train_model(self, i, previous_train):
        if previous_train:
            if i == 1:
                epochs = int(self.epochs / 4)
                acc_threshold = self.lstm_dict['max_accuracy'] + .01
                self.model.optimizer.learning_rate.assign(self.lstm_dict['adam_optimizer'] / 5)
            else:
                epochs = int(self.epochs / 4)
                acc_threshold = self.lstm_dict['max_accuracy'] + .01
                self.model.optimizer.learning_rate.assign(self.lstm_dict['adam_optimizer'] / 5)
        else:
            epochs = self.epochs
            acc_threshold = self.max_acc

        lr_scheduler = ReduceLROnPlateau(monitor='wl_class_loss',
                                         factor=0.85,
                                         patience=2,
                                         min_lr=.00000025,
                                         cooldown=2,
                                         verbose=2)
        self.model_plot = LivePlotLosses(plot_live=self.lstm_dict['plot_live'])

        train_data_gen, test_data_gen = self.get_input_datasets()

        stop_at_accuracy = StopAtAccuracy(accuracy_threshold=acc_threshold)

        self.model.fit(train_data_gen,
                       epochs=epochs,
                       verbose=1,
                       validation_data=test_data_gen,
                       callbacks=[lr_scheduler, self.model_plot, stop_at_accuracy],
                       shuffle=False)
        self.model_plot.save_plot(self.save_handler.data_folder, self.param)

    def get_model_summary_df(self, printft=False):
        if printft:
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

    def get_generator(self, traintf=True):
        if traintf:
            generator = CustomDataGenerator(self.mkt_data, self, self.batch_s)
        else:
            generator = CustomDataGenerator(self.mkt_data, self, self.batch_s, train=False)

        def generator_function():
            for i in range(len(generator)):
                yield generator[i]

        return generator_function

    def get_input_datasets(self):
        daily_shape = len(self.mkt_data.dailydata.columns) - 1
        intraday_shape = len(self.mkt_data.intradata.columns) - 1

        output_signature = (
            (tf.TensorSpec(shape=(None, self.daily_len, daily_shape), dtype=tf.float32),
             tf.TensorSpec(shape=(None, self.intra_len, intraday_shape), dtype=tf.float32)),
            {'wl_class': tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
             'pnl': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)})

        train_gen = self.get_generator(traintf=True)
        test_gen = self.get_generator(traintf=False)

        train_dataset = tf.data.Dataset.from_generator(
            train_gen,
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_generator(
            test_gen,
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        return train_dataset, test_dataset


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
        start_ind = index * self.batch_size
        end_ind = (index + 1) * self.batch_size
        train_inds = self.sample_ind_list[start_ind:end_ind]
        batch_gen = self.mkt_data.create_batch_input(train_inds,
                                                     self.lstm_model.daily_len,
                                                     self.train_tf)

        (x_day_arr, x_intra_arr), labels = next(batch_gen)
        y_wl_arr = labels['wl_class']  # Ensure shape is (None,)
        y_pnl_arr = labels['pnl']  # Ensure shape is (None,)

        return (x_day_arr, x_intra_arr), {'wl_class': y_wl_arr, 'pnl': y_pnl_arr}

    def set_attributes(self):
        if self.train_tf:
            self.sample_ind_list = list(range(len(self.mkt_data.y_train_pnl_df.index)))
            self.n_samples = len(self.mkt_data.y_train_pnl_df)

        else:
            self.sample_ind_list = list(range(len(self.mkt_data.y_test_pnl_df.index)))
            self.n_samples = len(self.mkt_data.y_test_pnl_df)

    def on_epoch_end(self):
        self.set_attributes()


class LivePlotLosses(Callback):
    def __init__(self, plot_live):
        super(LivePlotLosses, self).__init__()
        self.plot_live = plot_live
        self.epochs = []

        self.losses = []
        self.wl_class_losses = []
        self.pnl_losses = []
        self.auc_loss = []
        self.wl_class_accs = []
        self.wl_class_accs2 = []

        self.losses_val = []
        self.wl_class_losses_val = []
        self.pnl_losses_val = []
        self.auc_loss_val = []
        self.wl_class_accs_val = []
        self.wl_class_accs_val2 = []

        self.train_loss_line = None
        self.val_loss_line = None
        self.pnl_loss_line = None
        self.val_pnl_loss_line = None
        self.auc_loss_line = None
        self.val_auc_loss_line = None
        self.class_loss_line = None
        self.val_class_loss_line = None
        self.class_npv_line = None
        self.val_class_npv_line = None
        self.class_ppv_line = None
        self.val_class_ppv_line = None

        if self.plot_live:
            plt.ion()
        self.fig, self.axs = plt.subplots(2, 3, figsize=(10, 6))  # Create a 2x2 grid of subplots
        self.fig.tight_layout()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs.append(epoch + 1)  # Accumulate epochs
        self.losses.append(logs.get('loss'))
        self.wl_class_losses.append(logs.get('wl_class_loss'))
        self.pnl_losses.append(logs.get('pnl_loss'))
        self.auc_loss.append(logs.get('wl_class_auc_loss'))
        self.wl_class_accs.append(logs.get('wl_class_npv'))
        self.wl_class_accs2.append(logs.get('wl_class_ppv'))

        self.losses_val.append(logs.get('val_loss'))
        self.wl_class_losses_val.append(logs.get('val_wl_class_loss'))
        self.pnl_losses_val.append(logs.get('val_pnl_loss'))
        self.auc_loss_val.append(logs.get('val_wl_class_auc_loss'))
        self.wl_class_accs_val.append(logs.get('val_wl_class_npv'))
        self.wl_class_accs_val2.append(logs.get('val_wl_class_ppv'))

        # Clear previous plots
        # for ax in self.axs.flat:
        #     ax.clear()

        if self.train_loss_line is None:
            self.train_loss_line, = self.axs[0, 0].plot([], [], label="Train", marker='.', color='blue')
            self.val_loss_line, = self.axs[0, 0].plot([], [], label="Val", marker='.', color='darkred')
            self.axs[0, 0].set_title("Total Loss")
            self.axs[0, 0].legend()

            self.pnl_loss_line, = self.axs[0, 1].plot([], [], label="Train", marker='.', color='blue')
            self.val_pnl_loss_line, = self.axs[0, 1].plot([], [], label="Val", marker='.', color='darkred')
            self.axs[0, 1].set_title("PnL MSE")

            self.auc_loss_line, = self.axs[0, 2].plot([], [], label="Train", marker='.', color='blue')
            self.val_auc_loss_line, = self.axs[0, 2].plot([], [], label="Val", marker='.', color='darkred')
            self.axs[0, 2].set_title("WL Auc Loss")

            self.class_loss_line, = self.axs[1, 0].plot([], [], label="Train", marker='.', color='blue')
            self.val_class_loss_line, = self.axs[1, 0].plot([], [], label="Val", marker='.', color='darkred')
            self.axs[1, 0].set_title("WL Class Loss")

            self.class_npv_line, = self.axs[1, 1].plot([], [], label="Train", marker='.', color='blue')
            self.val_class_npv_line, = self.axs[1, 1].plot([], [], label="Val", marker='.', color='darkred')
            self.axs[1, 1].set_title("WL Class NPV")

            self.class_ppv_line, = self.axs[1, 2].plot([], [], label="Train", marker='.', color='blue')
            self.val_class_ppv_line, = self.axs[1, 2].plot([], [], label="Val", marker='.', color='darkred')
            self.axs[1, 2].set_title("WL Class PPV")

        self.train_loss_line.set_data(self.epochs, self.losses)
        self.val_loss_line.set_data(self.epochs, self.losses_val)

        self.pnl_loss_line.set_data(self.epochs, self.pnl_losses)
        self.val_pnl_loss_line.set_data(self.epochs, self.pnl_losses_val)

        self.auc_loss_line.set_data(self.epochs, self.auc_loss)
        self.val_auc_loss_line.set_data(self.epochs, self.auc_loss_val)

        self.class_loss_line.set_data(self.epochs, self.wl_class_losses)
        self.val_class_loss_line.set_data(self.epochs, self.wl_class_losses_val)

        self.class_npv_line.set_data(self.epochs, self.wl_class_accs)
        self.val_class_npv_line.set_data(self.epochs, self.wl_class_accs_val)

        self.class_ppv_line.set_data(self.epochs, self.wl_class_accs2)
        self.val_class_ppv_line.set_data(self.epochs, self.wl_class_accs_val2)

        for r in range(2):
            for c in range(3):
                self.axs[r, c].relim()
                self.axs[r, c].autoscale_view()

        if self.plot_live:
            self.fig.canvas.draw()
            plt.pause(0.2)

    def save_plot(self, save_loc, param_id):
        plt.savefig(f'{save_loc}\\param_{param_id}_plot.png', dpi=500)

    def on_train_end(self, logs=None):
        if self.plot_live:
            plt.ioff()  # Turn off interactive mode at the end
            plt.close()


class StopAtAccuracy(Callback):
    def __init__(self, accuracy_threshold=.99):
        super(StopAtAccuracy, self).__init__()
        self.accuracy_threshold = accuracy_threshold

    def on_epoch_end(self, epoch, logs=None):
        wl_ppv = logs.get('wl_class_ppv')
        wl_npv = logs.get('wl_class_ppv')
        if ((wl_ppv is not None and wl_ppv >= self.accuracy_threshold) and
                (wl_npv is not None and wl_npv >= self.accuracy_threshold)):
            print(f"\nReached {self.accuracy_threshold*100}% PPV & NPV accuracy, stopping training!")
            self.model.stop_training = True


class TemperatureScalingLayer(tf.keras.layers.Layer):
    def __init__(self, initial_temp=1.0, **kwargs):
        super(TemperatureScalingLayer, self).__init__(**kwargs)
        self.temperature = tf.Variable(initial_temp, trainable=True, dtype=tf.float32)
        self.initial_temp = initial_temp

    def call(self, logits):
        return logits / self.temperature

    def get_config(self):
        # Base config
        config = super().get_config()
        # Add custom argument
        config.update({
            "initial_temp": self.initial_temp
        })
        return config

