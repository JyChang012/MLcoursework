from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB
import pandas as pd
import numpy as np
from task import task1_load_all
from sklearn.utils import shuffle
import utils_task1 as utl
from matplotlib import pyplot as plt


def task1(repeat=10):
    mails_list = {'ham': task1_load_all('./data/贝叶斯模型编程/task1/ham/'),
                  'spam': task1_load_all('./data/贝叶斯模型编程/task1/spam/')}

    unique_words = utl.create_vocab_list(mails_list['ham'] + mails_list['spam'])

    design_mat = pd.DataFrame([], columns=unique_words + ['c'])

    name = 0
    for category in ('ham', 'spam'):
        for mail in mails_list[category]:
            counts = mail.value_counts(sort=False)
            counts = counts.rename(name)
            design_mat = design_mat.append(counts)
            design_mat.loc[name, 'c'] = category
            name += 1

    design_mat = design_mat.fillna(0)

    len_of_fold = int(design_mat.shape[0] / 5)

    accuracies = []
    nbg = MultinomialNB()
    acc = []
    for _ in range(repeat):
        for i in range(5):
            design_mat = shuffle(design_mat)
            test_set = design_mat.iloc[i * len_of_fold:(i + 1) * len_of_fold]
            train_set = design_mat.loc[design_mat.index.drop(test_set.index)]
            nbg.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
            y_est = nbg.predict(test_set.iloc[:, :-1])
            if test_set.loc[y_est != test_set.iloc[:, -1]].empty:
                print(f'Iteration {i}, no misclassified.')
            else:
                print(f'Iteration {i}, misclassified:\n', test_set.loc[y_est != test_set.iloc[:, -1]])
            accuracies.append(np.sum(y_est == test_set.iloc[:, -1]) / y_est.shape[0])

        avg = np.average(accuracies)
        print(f'avg acc = {avg}')
        acc.append(avg)

    print(f'Repeat {repeat} time, avg acc = {np.average(acc)}')
    return np.average(acc)
    pass


if __name__ == '__main__':
    task1()
