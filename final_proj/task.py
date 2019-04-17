import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

keras = tf.keras


def main():
    # load the data.
    train_data = pd.read_csv('./data/dataAll.csv', index_col=None)
    test_featues = pd.read_csv('./data/testAll.csv', index_col=None)

    train_data = shuffle(train_data)

    train_features = train_data.iloc[:, :-2]  # (3863, 30)
    train_labels = train_data.iloc[:, -2:]  # (3863, 2)

    model = keras.Sequential()
    model.add(tf.layers.Dense(40, activation='relu', kernel_regularizer='l2', bias_regularizer='l2', input_shape=(30,)))

    for width in [40] * 8 + [30, 30, 20, 20, 10, 8, 4]:
        model.add(tf.layers.Dense(width, activation='relu', kernel_regularizer='l2',
                                  bias_regularizer='l2'))

    model.add(tf.layers.Dense(2))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    model.fit(train_features.values, train_labels.values, epochs=40, batch_size=32, validation_split=0.2)
    pass


if __name__ == '__main__':
    main()
    pass
