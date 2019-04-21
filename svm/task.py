from task1 import SVM_Functions as svm
import numpy as np
import pandas as pd
from scipy.io import loadmat


def task2():
    x, y = svm.loadData('./task2/task2.mat')
    x = np.array(x)
    y = np.array(y)

    record = pd.DataFrame(columns=pd.Series([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], name='sigma'),
                          index=pd.Series([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], name='C'))

    for c in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30):
        for sigma in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30):
            K_matrix = svm.gaussianKernel(x, sigma=sigma)
            model = svm.svmTrain_SMO(x, y, C=c, kernelFunction='gaussian', K_matrix=K_matrix)
            y_est = svm.svmPredict(model, x, sigma)
            acc = np.sum((y_est == y)) / y_est.shape[0]
            record.loc[c, sigma] = acc

    record.to_csv('./task2_record')


def task3():
    x_train, y_train = svm.loadData('./task3/task3_train.mat')
    x_test = loadmat('./task3/task3_test.mat')['X']

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)

    print(f'train: （#samples，#features） = {x_train.shape}\ntest: （#samples，#features） = {x_test.shape}')

    # pre train the model
    # l = int(0.8 * x_train.shape[0])

    # x_train_tr, y_train_tr = x_train[:l, ], y_train[:l, ]
    # x_train_te, y_train_te = x_train[l:, ], y_train[l:, ]

    # model = svm.svmTrain_SMO(x_train_tr, y_train_tr, C=30)

    # y_train_tr_est = svm.svmPredict(model, x_train_tr)
    # print(f'in sample err: {np.sum((y_train_tr_est == y_train_tr)) / y_train_tr_est.shape[0]}')

    # y_train_te_est = svm.svmPredict(model, x_train_te)
    # print(f'outta sample err: {np.sum((y_train_te_est == y_train_te)) / y_train_te_est.shape[0]}')

    # training
    model = svm.svmTrain_SMO(x_train, y_train, C=30)
    y_test_est = svm.svmPredict(model, x_test)

    print(y_test_est)
    np.savetxt('./task3_result.txt', y_test_est, fmt='%1d')
    pass


if __name__ == '__main__':
    task3()
