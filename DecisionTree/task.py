import numpy as np
import pandas as pd

from decision_tree import DecisionTree


def task1():
    data = pd.read_csv('./data/Task1/lenses.txt', sep='\t', header=None)
    tree = DecisionTree(data, prune_rate=0.2, prune_type='pre')
    result = tree.discriminate(data.iloc[:, :-1])
    print('\n', result == data.iloc[:, -1])
    in_sample_err = 1 - np.sum(result == data.iloc[:, -1]) / result.shape[0]
    print(f'In sample error rate: {in_sample_err}')
    tree.visualize()


def pre_process_task2(raw_data=pd.DataFrame()):
    train_data = raw_data.iloc[:, [0, 1, 3]]
    BSSID_col = list(train_data.index.unique())
    idx = train_data.finLabel.unique()
    design_mat = pd.DataFrame(columns=BSSID_col + ['RoomLabel'], index=idx)
    for fin in train_data.finLabel.unique():
        design_mat.loc[fin, :-1] = train_data.iloc[list(train_data.loc[:, 'finLabel'] == fin), 0]
        design_mat.loc[fin, 'RoomLabel'] = train_data.iloc[list(train_data.loc[:, 'finLabel'] == fin), 1].iloc[0]
    design_mat = design_mat.fillna(-500)
    return design_mat


def task2():
    raw_train_data = pd.read_csv('./data/Task2/TrainDT.csv', index_col=0)
    raw_test_data = pd.read_csv('./data/Task2/TestDT.csv', index_col=0)

    train_design_mat = pre_process_task2(raw_train_data)
    test_design_mat = pre_process_task2(raw_test_data)

    tree = DecisionTree(train_design_mat, continuous_col=train_design_mat.columns[:-1], type='ID3', prune_rate=None,
                        prune_type='pre')
    tree.visualize()

    result_in = tree.discriminate(train_design_mat.iloc[:, :-1])
    print('\n', result_in == train_design_mat.iloc[:, -1])
    in_sample_err = 1 - np.sum(result_in == train_design_mat.iloc[:, -1]) / result_in.shape[0]
    print(f'In sample error rate: {in_sample_err}')

    result_test = tree.discriminate(test_design_mat.iloc[:, :-1])

    print('\n', result_test == test_design_mat.iloc[:, -1])
    outta_sample_err = 1 - np.sum(result_test == test_design_mat.iloc[:, -1]) / result_test.shape[0]
    print(f'Out of sample error rate: {outta_sample_err}')
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
    # train_data = pd.read_csv('./data/Task3/train/train_data.txt', header=None)
    # train_labels = pd.read_csv('./data/Task3/train/train_labels.txt', sep=' ', header=None).iloc[:, 0]
    # test_data = pd.read_csv('./data/Task3/test/test_data.txt', header=None)
    # train_design_mat = pre_process_task3(train_data)
    # train_design_mat['label'] = train_labels
    # test_design_mat = pre_process_task3(test_data)
    # train_design_mat.to_csv('train_design_mat.csv')
    # test_design_mat.to_csv('test_design_mat.csv')

    # read data
    train_design_mat = pd.read_csv('train_design_mat.csv', index_col=0)
    test_design_mat = pd.read_csv('test_design_mat.csv', index_col=0)

    # continuous
    # tree = DecisionTree(train_design_mat, type='ID3', continous_col=train_design_mat.columns[:-1])
    # discrete
    train_design_mat = (train_design_mat > 0)
    test_design_mat = (test_design_mat > 0)
    tree = DecisionTree(train_design_mat, type='ID3')
    tree.visualize()

    result_in = tree.discriminate(train_design_mat.iloc[:, :-1])
    in_sample_err = 1 - np.sum(result_in == train_design_mat.iloc[:, -1]) / result_in.shape[0]
    print(f'In sample error rate: {in_sample_err}')

    result_test = tree.discriminate(test_design_mat)
    result_test.to_csv('result_test.csv')
    pass


if __name__ == '__main__':
    task2()
