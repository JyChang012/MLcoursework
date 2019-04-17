import pandas as pd

from decision_tree import DecisionTree


def hw2_2():
    data = pd.DataFrame({'年龄': ['青年'] * 5 + ['中年'] * 5 + ['老年'] * 5, '有工作': list('nnyynnnynnnnyyn'),
                         '房子': list('nnnynnnyyyyynnn'), '信贷': list('nggnnnggeeeggen'),
                         '标签': list('nnyynnnyyyyyyyn')})
    Dtree = DecisionTree(data)
    print(Dtree.discriminate(data.iloc[:, :-1]) == data.iloc[:, -1])
    pass


def hw2_4():
    data = pd.DataFrame({'gender': list('mmmmmmffffmmmmffffff'), 'car_model': list('hsssssssslhhhlllllll'),
                         'clothes_size': list('smmleessmllemessmmml'), 'label': [0] * 10 + [1] * 10})
    Dtree = DecisionTree(data, type='CART')
    print(Dtree.discriminate(data.iloc[:, :-1]) == data.iloc[:, -1])
