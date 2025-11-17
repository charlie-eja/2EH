import os
import pandas as pd
import numpy as np
import tensorflow as tf
from utilz import read_data, read_generated_data, split_series, moving_window


def DPAS_data_loader(type_list, runs, n_predict_features, te=False, n_past=1, n_future=1, std=True, d_n=False):
    df = pd.read_excel(os.path.join(os.getcwd(), 'data', 'attribute_matrix.xlsx'), sheet_name=1)
    x_total, y_total, dec_x_total, attr_total = np.empty([0, n_past, 52]), np.empty(
        [0, n_future, n_predict_features]), np.empty(
        [0, n_future, 52-n_predict_features]), np.empty([0, n_future, df.shape[1] - 1])
    n_attr = np.zeros([df.shape[1] - 1])

    for fault_type_num in type_list:

        if fault_type_num == 0:
            f_attr = n_attr.astype(int)
        else:
            f_attr = df.loc[df['F/Att'] == 'F' + str(fault_type_num)].to_numpy()[:, 1:][0]

        for i in runs:
            data = read_data(fault_type_num, te, i, std, d_n)
            x, y, attr = split_series(data, n_past, n_future, n_attr, f_attr, 20)
            n_features = x.shape[2]
            x = reshape_feature(x, n_features)
            y = reshape_feature(y, n_features)
            dec_x = y[:, :, n_predict_features:len(y)]
            y = y[:, :, 0:n_predict_features]
            x_total = np.vstack([x_total, x])
            y_total = np.vstack([y_total, y])
            dec_x_total = np.vstack([dec_x_total, dec_x])
            attr_total = np.vstack([attr_total, attr])

    return x_total, attr_total.astype('float32'), y_total, dec_x_total


def reshape_feature(x, n_features):
    return x.reshape((x.shape[0], x.shape[1], n_features))


def classifier_data_loader(known_fault_list, fault_list, te=False, n_run=10, shuffle=True, more_normal_data=True,
                           timesteps=20, step=1, unknown_list=None, val=False):

    data_all = []
    label_all = []
    tag = ''.join([str(num) for num in known_fault_list])
    unknown_shift_idx = 0

    if unknown_list:
        label_len = len(fault_list) - len(list(set(fault_list) & set(unknown_list))) + 1
    else:
        label_len = len(fault_list)

    for idx, fault_type in enumerate(fault_list):

        if not fault_type and fault_type in known_fault_list and more_normal_data:
            n_run_ = n_run*4
        else:
            n_run_ = n_run

        if unknown_list:
            if fault_type in unknown_list:
                unknown_shift_idx += 1

        for i in range(n_run_):

            if fault_type not in known_fault_list:
                if te and not val:
                    data = read_data(fault_type, te, i + 1)
                else:
                    data = read_generated_data(tag, fault_type, i)
            else:
                data = read_data(fault_type, te, i+1)

            mw_data = moving_window(data, timesteps, step)

            if unknown_list:
                if fault_type in unknown_list:
                    label = tf.one_hot([label_len-1]*mw_data.shape[0], label_len)

                else:
                    label = tf.one_hot([idx-unknown_shift_idx]*mw_data.shape[0], label_len)
            else:
                label = tf.one_hot([idx] * mw_data.shape[0], label_len)

            if len(data_all):
                data_all = np.vstack((data_all, mw_data))
                label_all = np.vstack((label_all, label))
            else:
                data_all = mw_data
                label_all = label

    dataset = tf.data.Dataset.from_tensor_slices((data_all, label_all))
    if shuffle:
        dataset = dataset.shuffle(1000).batch(32)
        return dataset
    return [data_all, label_all]


if __name__ == '__main__':
    known_faults = [0, 2, 12, 14]
    faults = [0, 2, 12, 13, 14]
    train_dataset = classifier_data_loader(known_faults, faults, n_run=10, shuffle=False)
    val_dataset = classifier_data_loader(known_faults, faults, te=True, n_run=1, shuffle=False)
    print()



import tensorflow as tf


class DPAS_seq2seq(tf.keras.Model):
    def __init__(self, n_predict_features, attr_en):
        super().__init__()
        self.attr_en = attr_en
        self.enc = tf.keras.layers.LSTM(56, return_state=True)
        self.dec = tf.keras.layers.LSTM(56, return_sequences=True)
        self.td = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_predict_features))

    def call(self, inputs):
        enc_input, dec_attr, dec_x = inputs

        if self.attr_en:
            dec_input = tf.keras.layers.Concatenate(axis=2)([dec_attr, dec_x])
        else:
            dec_input = dec_x

        enc_out = self.enc(enc_input)
        enc_stat = enc_out[1:]

        dec_out = self.dec(dec_input, initial_state=enc_stat)
        outputs = self.td(dec_out)

        return outputs


