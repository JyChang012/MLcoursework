import pandas as pd
import numpy as np
import os
import utils_task1 as utl
from sklearn.utils import shuffle
import naive_bayes as nb
from matplotlib import pyplot as plt


def task1_load_all(path=''):
    file_list = os.listdir(path)
    emails_list = []
    for file_name in file_list:
        if file_name.endswith('.txt'):
            with open(path+file_name, 'r') as f:
                read_file = f.read()
            words_list = pd.Series(utl.text_parse(read_file))
            emails_list.append(words_list)
    return emails_list


def task1():
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
    design_mat = shuffle(design_mat)
    len_of_fold = int(design_mat.shape[0] / 5)
    accuracies = []

    nb_classifier = nb.NaiveBayes(laplacian_correction=False, continuous_col=design_mat.columns[:-1])
    for i in range(5):
        test_set = design_mat.iloc[i * len_of_fold:(i + 1) * len_of_fold]
        train_set = design_mat.loc[design_mat.index.drop(test_set.index)]
        nb_classifier.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
        y_est = nb_classifier.discriminate(test_set.iloc[:, :-1])
        print(f'Iteration {i}, misclassified:\n', test_set.loc[y_est != test_set.iloc[:, -1]])
        accuracies.append(np.sum(y_est == test_set.iloc[:, -1]) / y_est.shape[0])

    print(f'avg. acc = {np.average(accuracies)}')
    plt.plot(accuracies)
    plt.title('accuracy')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.show()
    pass


if __name__ == '__main__':
    task1()
