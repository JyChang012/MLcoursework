from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer

keras = tf.keras


def task1():
    train_data = pd.read_csv('./Data/Task1/horseColicTraining.txt', sep='\t', header=None)
    test_data = pd.read_csv('./Data/Task1/horseColicTest.txt', sep='\t', header=None)

    train_features = train_data.iloc[:, :-1]
    train_labels = train_data.iloc[:, -1]

    cls = LogisticRegression(C=2)

    cls.fit(train_features, train_labels)
    score = cls.score(test_data.iloc[:, :-1], test_data.iloc[:, -1])
    print(score)


def task2(max_features=10000, depth=30):
    with open('./Data/Task2/train/train_texts.dat', 'rb') as f:
        train_text = pk.load(f)

    with open('./Data/Task2/test/test_texts.dat', 'rb') as f:
        test_text = pk.load(f)

    vectorizer = TfidfVectorizer(max_features=max_features)
    train_features = vectorizer.fit_transform(train_text)  # (11314, 10000)
    test_features = vectorizer.transform(test_text)  # (7532, 10000)
    train_labels = pd.read_csv('./Data/Task2/train/train_labels.txt', header=None).values  # 20 unique classes

    train_labels = train_labels[:20, ]
    train_features = train_features[:20, ]

    model = keras.Sequential()
    model.add(tf.layers.Dense(max_features, activation='relu', kernel_regularizer='l2', input_shape=(max_features,)))

    for width in np.linspace(max_features, 20, depth).astype(np.int):
        model.add(tf.layers.Dense(width, activation='relu', kernel_regularizer='l2'))

    model.add(tf.layers.Dense(20, activation='softmax', kernel_regularizer='l2'))

    model.compile(optimizer='adam', loss=tf.losses.log_loss, metrics=['accuracy', tf.losses.log_loss])
    model.fit(train_features, train_labels, batch_size=32, epochs=200, validation_split=0.2)
    pass


if __name__ == '__main__':
    task2(20, 30)
