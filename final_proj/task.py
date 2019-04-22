import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import os
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

keras = tf.keras

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    # plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    # plt.ylim([0, 20])
    plt.legend()
    plt.show()


def main():
    # load the data.
    train_data = pd.read_csv('./data/dataAll.csv', index_col=None)
    test_featues = pd.read_csv('./data/testAll.csv', index_col=None)

    attrs = []

    for band in ('21', '35'):
        for sector in np.array([1, 3, 8, 9, 13, 15]) - 1:
            attrs.append(band + '00' + str(sector))

    train_data = shuffle(train_data)

    train_features = train_data.iloc[:, :-2]
    train_features = train_features.loc[:, attrs]  # (3863, 30)
    train_labels = train_data.iloc[:, -2:]  # (3863, 2)

    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(12, activation='relu',
                                    input_shape=(12,)))

    for width in [12] * 32:  # 32 or 36
        model.add(tf.keras.layers.Dense(width, activation='relu'))

    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics=['mean_absolute_error',
                                                                                            'mean_squared_error'])

    callbacks = [keras.callbacks.EarlyStopping(patience=20), PrintDot()]

    print(model.summary())
    history = model.fit(train_features.values, train_labels.values, epochs=20000, batch_size=32,
                        validation_split=0.2, callbacks=callbacks)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plot_history(history)
    print(model.predict(train_features, batch_size=32))
    pass


def main_full():
    # load the data.
    train_data = pd.read_csv('./data/dataAll.csv', index_col=None)
    test_featues = pd.read_csv('./data/testAll.csv', index_col=None)

    train_data = shuffle(train_data)

    train_features = train_data.iloc[:, :-2]
    train_labels = train_data.iloc[:, -2:]  # (3863, 2)

    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(30, activation='relu', kernel_regularizer=keras.regularizers.l2(),
                                    input_shape=(30,)))

    for width in [30] * 32:  # 32 or 36
        model.add(tf.keras.layers.Dense(width, activation='relu', kernel_regularizer=keras.regularizers.l2()))

    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics=['mean_absolute_error',
                                                                                            'mean_squared_error'])

    callbacks = [keras.callbacks.EarlyStopping(patience=5), PrintDot()]

    print(model.summary())
    history = model.fit(train_features.values, train_labels.values, epochs=20000, batch_size=32,
                        validation_split=0.2, callbacks=callbacks)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plot_history(history)
    print(model.predict(train_features, batch_size=32))
    pass


def main_post():
    # load the data.
    train_data = pd.read_csv('./data/dataAll.csv', index_col=None)
    test_features = pd.read_csv('./data/testAll.csv', index_col=None)

    attrs = []

    for band in ('21', '35'):
        for sector in np.array([1, 3, 8, 9, 13, 15]) - 1:
            attrs.append(band + '00' + str(sector))

    train_features = train_data.iloc[:, :-2]
    train_features = train_features.loc[:, attrs]  # (3863, 30)
    train_labels = train_data.iloc[:, -2:]  # (3863, 2)

    test_features = test_features.loc[:, attrs]

    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(12, activation='relu', kernel_regularizer=keras.regularizers.l2(),
                                    input_shape=(12,)))

    for width in [12] * 32:  # 32 or 36
        model.add(tf.keras.layers.Dense(width, activation='relu', kernel_regularizer=keras.regularizers.l2()))

    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics=['mean_absolute_error',
                                                                                            'mean_squared_error'])

    print(model.summary())
    model.fit(train_features.values, train_labels.values, epochs=75, batch_size=32)  # 70 opt

    result = model.predict(test_features, batch_size=32)
    result = pd.DataFrame(result, columns=['x', 'y'], index=pd.Series(list(range(result.shape[0])), name='id'))
    result.to_csv('/home/changjy/pycharm_mapping/pycharm_project_148/final_proj/3.csv')


if __name__ == '__main__':
    main()
    pass
