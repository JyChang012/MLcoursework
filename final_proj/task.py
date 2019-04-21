import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

keras = tf.keras

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4


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
    model.add(tf.keras.layers.Dense(12, activation='relu', kernel_regularizer='l2', input_shape=(12,)))

    for width in [12] * 32 + [int(i) for i in np.linspace(12, 2, 32)]:
        model.add(tf.keras.layers.Dense(width, activation='relu', kernel_regularizer='l2'))
        model.add(keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=[])

    callbacks = [keras.callbacks.EarlyStopping(patience=50)]

    with tf.Session(config=config) as sess:
        sess.run(model.fit(train_features.values, train_labels.values, epochs=20000, batch_size=32,
                           validation_split=0.2, callbacks=callbacks))
    pass


if __name__ == '__main__':
    main()
    pass
