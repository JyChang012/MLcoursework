import pandas as pd
import numpy as np
import os
import utils_task1 as utl
from sklearn.utils import shuffle
import naive_bayes as nb
from matplotlib import pyplot as plt
from Bayesian.data.贝叶斯模型编程.task2.NBC import score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.feature_extraction.text import TfidfTransformer
import jieba as jb
import os


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



def pre_process_task3(data):
    data = data[0].str.split(' ', expand=True)
    data_val = data.values
    for row in range(data_val.shape[0]):
        data_val[row, list(data_val[row,] == None)] = 0
    data_val = data_val.astype(np.int)
    data = pd.DataFrame(data_val)
    design_mat = pd.DataFrame(np.zeros((data.shape[0], 10000)), columns=list(range(1, 10001)))
    for row in data.index:
        design_mat.loc[row] = data.loc[row].value_counts(sort=False)
    return design_mat.fillna(0)


def task3():
    # prepare data
    # train_data = pd.read_csv('./data/贝叶斯模型编程/Task3/train/train_data.txt', header=None)
    # train_labels = pd.read_csv('./data/贝叶斯模型编程/Task3/train/train_labels.txt', sep=' ', header=None).iloc[:, 0]
    # test_data = pd.read_csv('./data/贝叶斯模型编程/Task3/test/test_data.txt', header=None)
    # train_design_mat = pre_process_task3(train_data)
    # train_design_mat['label'] = train_labels
    # test_design_mat = pre_process_task3(test_data)
    # train_design_mat.to_pickle('train_design_mat.pkl')
    # test_design_mat.to_pickle('test_design_mat.pkl')

    train_design_mat = pd.read_pickle('train_design_mat.pkl')
    test_design_mat = pd.read_pickle('test_design_mat.pkl')

    # acc = {'gaussian': [], 'complement': [], 'multi': []}
    train_len = int(0.8 * train_design_mat.shape[0])
    train_design_mat = shuffle(train_design_mat)

    # Use tfidf transform
    trans = TfidfTransformer(sublinear_tf=True)
    label = train_design_mat['label']
    train_design_mat_trans = trans.fit_transform(train_design_mat.iloc[:, :-1])
    train_design_mat_trans = pd.DataFrame(train_design_mat_trans.toarray(), index=train_design_mat.index,
                                    columns=train_design_mat.columns[:-1])
    train_design_mat_trans['label'] = label

    train_set = train_design_mat_trans.iloc[:train_len]
    test_set = train_design_mat_trans.iloc[train_len:]

    acc_afidf = []
    alpha_range = np.linspace(0, 1, 10)
    for alpha in alpha_range:
        classifier = MultinomialNB(alpha=alpha)
        classifier.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
        acc_afidf.append(classifier.score(test_set.iloc[:, :-1], test_set.iloc[:, -1]))

    # without using tf_idf
    acc = []
    train_set = train_design_mat.iloc[:train_len]
    test_set = train_design_mat.iloc[train_len:]

    for alpha in alpha_range:
        classifier = MultinomialNB(alpha=alpha)
        classifier.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
        acc.append(classifier.score(test_set.iloc[:, :-1], test_set.iloc[:, -1]))

    plt.plot(alpha_range, acc_afidf, marker='o', label='using tf-idf with sublinear tf')
    plt.plot(alpha_range, acc, marker='x', label='without using tf-idf')
    plt.title('accuracy vs smoothing parameter alpha')
    plt.legend()
    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.show()
    pass


def task3_predict():
    train_design_mat = pd.read_pickle('train_design_mat.pkl')
    test_design_mat = pd.read_pickle('test_design_mat.pkl')

    label = train_design_mat['label']

    len_train = train_design_mat.shape[0]

    trans = TfidfTransformer(sublinear_tf=True)
    all_data = train_design_mat.iloc[:, :-1].append(test_design_mat)
    all_data = trans.fit_transform(all_data)
    all_data = pd.DataFrame(all_data.toarray(), index=list(train_design_mat.index)+list(test_design_mat.index),
                                          columns=train_design_mat.columns[:-1])

    train_design_mat = all_data.iloc[:len_train]
    test_design_mat = all_data.iloc[len_train:]

    alpha_range = np.linspace(0, 1, 10)
    for i, alpha in zip(range(1, 10), alpha_range):
        classifier = MultinomialNB(alpha=alpha)
        classifier.fit(train_design_mat, label)
        result = classifier.predict(test_design_mat)
        np.savetxt(f'./畅嘉宇U201613382第4章贝叶斯网络/{i}.txt',  result, fmt='%1.d')
    pass


if __name__ == '__main__':
    task2_preprocess()
