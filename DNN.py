'''
@autor: Wei Shuai
Time: 2022.1.10 at ZJU
'''

import tensorflow as tf
from tensorflow_core import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

File_Path =r'D:\Desktop'
DIR = 'file_to_store'
readname = 'data.csv'

BATCHSZ = 50
EACH_EPOCH = 1000
LR = 0.001
SEED = 7
CELL = 256

def train_test():
    dataset = pd.read_csv(File_Path + '\\' + readname, header=0,  encoding='utf-8')
    values = dataset.values
    values = values.astype('float32')
    np.random.seed(SEED)
    np.random.shuffle(values)
    tf.random.set_seed(SEED)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scaled = 0.8*scaled + 0.2
    train = scaled[:24000, :]
    test = scaled[24000:, :]
    train_X = train[:, :4]
    train_y = train[:, 4:]
    test_X = test[:, :4]
    test_y = test[:, 4:]
    train_X = np.array(train_X, dtype=np.float32)
    train_y = np.array(train_y)
    test_X = np.array(test_X, dtype=np.float32)
    test_y = np.array(test_y)

    return train_X, train_y, test_X, test_y, scaler

def model_build(train_datas): #train_datas = train_X
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(CELL/2, input_shape=(train_datas.shape[1:]), activation='linear'))
    model.add(keras.layers.Dense(CELL, activation='relu'))
    model.add(keras.layers.Dense(CELL, activation='relu'))
    model.add(keras.layers.Dense(CELL, activation='relu'))
    model.add(keras.layers.Dense(CELL, activation='relu'))
    model.add(keras.layers.Dense(CELL/2, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softplus'))
    model.compile(optimizer=keras.optimizers.Adam(lr=LR, amsgrad=True), loss='mse', metrics=[rmse])  # mae: mean_absolute_error

    return model

def model_fit(model, train_datas, train_labels,x_test, y_test):
    checkpoint_save_path = "./{}/checkpoint{}/DNN_stock.ckpt".format(DIR, SEED)

    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    lr_reduce = keras.callbacks.ReduceLROnPlateau('val_loss',
                                                  patience=4,
                                                  factor=0.9,
                                                  min_lr=0.00001)
    best_model = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 mode='min',
                                                 )

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    history = model.fit(
        train_datas, train_labels,
        batch_size=BATCHSZ,
        epochs=EACH_EPOCH,
        verbose=2,
        validation_split=0.2,
        callbacks=[
            best_model,
            early_stop,
            lr_reduce,
        ]
    )
    model.save("./{}/checkpoint{}/model.h5".format(DIR, SEED))
    return model, history

def rmse(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1))

if __name__ == '__main__':
    train_X, train_y, test_X, test_y, scaler = train_test()
    model = model_build(train_X)
    model, history = model_fit(model, train_X, train_y, test_X, test_y)
