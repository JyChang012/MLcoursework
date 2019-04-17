import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


keras = tf.keras


def main():
    train_data = pd.read_csv('./data/dataAll.csv', index_col=None)
    test_featues = pd.read_csv('./data/testAll.csv', index_col=None)

    train_data = shuffle(train_data)

    train_features = train_data.iloc[:, :-2]  # (3863, 30)
    train_labels = train_data.iloc[:, -2:]  # (3863, 2)