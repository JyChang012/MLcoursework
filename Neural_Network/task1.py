import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def task1():
    train_data = pd.read_csv('./Data/Task1/horseColicTraining.txt', sep='\t', header=None)
    test_data = pd.read_csv('./Data/Task1/horseColicTest.txt', sep='\t', header=None)

    train_features = train_data.iloc[:, :-1]
    train_labels = train_data.iloc[:, -1]

    cls = LogisticRegression(C=2)

    cls.fit(train_features, train_labels)
    score = cls.score(test_data.iloc[:, :-1], test_data.iloc[:, -1])
    print(score)


if __name__ == '__main__':
    task1()
