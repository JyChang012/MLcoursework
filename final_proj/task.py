import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import os
from matplotlib import pyplot as plt
import sklearn.neighbors as neighbors
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

keras = tf.keras

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4


def read_data():
    train_data = pd.read_csv('./data/dataAll.csv', index_col=None)
    test_featues = pd.read_csv('./data/testAll.csv', index_col=None)
    return train_data, test_featues


def calculate_err(truth, est):
    return np.sum(np.apply_along_axis(np.linalg.norm, axis=1, arr=truth - est)) / \
          truth.shape[0]


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def cal_coverage(df):
    """calculate the coverage of the selected sectors."""
    rst = df > -105
    rst = rst.sum(axis=1) > 0
    cov = rst.sum() / rst.shape[0]
    # print(cov)
    return cov


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    # plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    # plt.ylim([0, 20])
    plt.legend()
    plt.show()


def main():
    """Find the optimal hyper parameter."""
    # load the data.
    train_data, test_features = read_data()
    attrs = pd.Index(['21000', '21002', '21003', '21005', '21006', '21007', '210012', '35004', '35008', '350014'])

    # for band in ('21', '35'):
    #     for sector in np.array([1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15]) - 1:
    #         attrs.append(band + '00' + str(sector))

    train_data = shuffle(train_data)

    train_features = train_data.iloc[:, :-2]

    # for col in train_features.columns:
    #     train_features.loc[train_features.loc[:, col] < -105, col] = -500

    train_features = train_features.loc[:, attrs]  # (3863, 30)
    train_labels = train_data.iloc[:, -2:]  # (3863, 2)

    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(attrs.shape[0], activation='relu',
                                    input_shape=(attrs.shape[0],)))

    for width in [attrs.shape[0]] * 32:  # 32 or 36
        model.add(tf.keras.layers.Dense(width, activation='relu'))

    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics=['mean_absolute_error',
                                                                                            'mean_squared_error'])

    callbacks = [keras.callbacks.EarlyStopping(patience=20), PrintDot()]

    print(model.summary())
    history = model.fit(train_features.values, train_labels.values, epochs=20000, batch_size=32,
                        validation_split=0.2, callbacks=callbacks)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plot_history(history)
    print(model.predict(train_features, batch_size=32))
    pass


def main_full():
    """No sector deleted."""
    # load the data.
    train_data, test_features = read_data()

    train_data = shuffle(train_data)

    train_features = train_data.iloc[:, :-2]

    for col in train_features.columns:
        train_features.loc[train_features.loc[:, col] < -105, col] = -500

    cal_coverage(train_features)
    train_labels = train_data.iloc[:, -2:]  # (3863, 2)

    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(30, activation='relu', kernel_regularizer=keras.regularizers.l2(),
                                    input_shape=(30,)))

    for width in [30] * 32:  # 32 or 36
        model.add(tf.keras.layers.Dense(width, activation='relu', kernel_regularizer=keras.regularizers.l2()))

    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics=['mean_absolute_error',
                                                                                            'mean_squared_error'])

    callbacks = [keras.callbacks.EarlyStopping(patience=5), PrintDot()]

    print(model.summary())
    history = model.fit(train_features.values, train_labels.values, epochs=20000, batch_size=32,
                        validation_split=0.2, callbacks=callbacks)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plot_history(history)
    print(model.predict(train_features, batch_size=32))
    pass


def main_post():
    """Use the estimated optimal hyper parameter."""
    # load the data.
    train_data, test_features = read_data()

    attrs = []

    for band in ('21', '35'):
        for sector in np.array([1, 3, 8, 9, 13, 15]) - 1:
            attrs.append(band + '00' + str(sector))

    train_features = train_data.iloc[:, :-2]
    train_features = train_features.loc[:, attrs]  # (3863, 30)
    train_labels = train_data.iloc[:, -2:]  # (3863, 2)

    test_features = test_features.loc[:, attrs]

    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(12, activation='relu', kernel_regularizer=keras.regularizers.l2(),
                                    input_shape=(12,)))

    for width in [12] * 32:  # 32 or 36
        model.add(tf.keras.layers.Dense(width, activation='relu', kernel_regularizer=keras.regularizers.l2()))

    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics=['mean_absolute_error',
                                                                                            'mean_squared_error'])

    print(model.summary())
    model.fit(train_features.values, train_labels.values, epochs=75, batch_size=32)  # 70 opt

    result = model.predict(test_features, batch_size=32)
    result = pd.DataFrame(result, columns=['x', 'y'], index=pd.Series(list(range(result.shape[0])), name='id'))
    result.to_csv('/home/changjy/pycharm_mapping/pycharm_project_148/final_proj/3.csv')


def knn_regressor():
    attrs = []
    for band in ('21', '35'):
        for sector in np.array([1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15]) - 1:
            attrs.append(band + '00' + str(sector))

    train_data, test_features = read_data()

    # train_data = train_data.loc[:, attrs + list(train_data.columns[-2:])]
    # test_features = test_features.loc[:, attrs]

    l = int(0.8 * train_data.shape[0])

    n = 20
    err = 0
    for _ in range(n):
        train_data = shuffle(train_data)

        train_data_tr = train_data.iloc[:l]
        train_data_te = train_data.iloc[l:]

        rgs = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance')
        rgs.fit(train_data_tr.iloc[:, :-2], train_data_tr.iloc[:, -2:])
        rst = rgs.predict(train_data_te.iloc[:, :-2])
        err += calculate_err(train_data_te.iloc[:, -2:].values, rst)

    print(err / n)


def visualize():
    attrs = pd.Index(['21000', '21002', '21003', '21005', '21006', '21007', '210012', '35004', '35008', '350014'])
    train_data, test_features = read_data()
    train_data = train_data.loc[:, attrs]
    cls = PCA(n_components=2)
    trans = cls.fit_transform(train_data.iloc[:, :-2])
    print(cls.singular_values_)
    plt.scatter(trans[:, 0], trans[:, 1], marker='x')
    plt.show()


def greedy_search_knn():
    train_data, test_features = read_data()
    l = int(0.8 * train_data.shape[0])

    attrs = test_features.columns

    rgs = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance')

    for sector_num in range(test_features.shape[1], 0, -1):
        dropped_attr2err = pd.Series(index=attrs)
        for dropped_attr in attrs:
            dropped_attrs = attrs[attrs != dropped_attr]
            coverage = cal_coverage(train_data.loc[:, dropped_attrs])
            if coverage < 0.95:
                dropped_attr2err.loc[dropped_attr] = np.inf
                continue
            else:
                err = 0
                n = 10
                for _ in range(n):
                    train_data = shuffle(train_data)
                    train_features_tr = train_data.loc[train_data.index[:l], dropped_attrs]
                    train_features_te = train_data.loc[train_data.index[l:], dropped_attrs]

                    train_labels_tr = train_data.iloc[:l, -2:]
                    train_labels_te = train_data.iloc[l:, -2:]

                    rgs.fit(train_features_tr, train_labels_tr)
                    rst = rgs.predict(train_features_te)
                    err += calculate_err(train_labels_te.values, rst)

                dropped_attr2err.loc[dropped_attr] = err / n
        if (dropped_attr2err == np.inf).all():
            break
        else:
            optml_attr = dropped_attr2err.idxmin()
            optml_err = dropped_attr2err.min()
            attrs = attrs.drop(optml_attr)

    print(f'sector num = {sector_num}')
    print(f'coverage = {cal_coverage(train_data.loc[:, attrs])}')
    print(f'optimal err = {optml_err}')
    print(attrs)
    pass

    train_features = train_data.loc[:, attrs]
    train_labels = train_data.iloc[:, -2:]

    rgs.fit(train_features, train_labels)
    rst = rgs.predict(test_features.loc[:, attrs])

    rst = pd.DataFrame(rst, columns=['x', 'y'], index=pd.Series(list(range(rst.shape[0])), name='id'))
    rst.to_csv('knn_new3.csv')


def greedy_search_dnn():
    train_data, test_features = read_data()
    l = int(0.8 * train_data.shape[0])

    attrs = test_features.columns
    callbacks = [keras.callbacks.EarlyStopping(patience=10)]

    for sector_num in range(test_features.shape[1], 0, -1):
        dropped_attr2err = pd.Series(index=attrs)
        for dropped_attr in attrs:
            dropped_attrs = attrs[attrs != dropped_attr]
            coverage = cal_coverage(train_data.loc[:, dropped_attrs])
            if coverage < 0.95:
                dropped_attr2err.loc[dropped_attr] = np.inf
                continue
            else:
                err = 0
                n = 10
                for _ in range(n):
                    train_data = shuffle(train_data)
                    train_features_tr = train_data.loc[train_data.index[:l], dropped_attrs]
                    train_features_te = train_data.loc[train_data.index[l:], dropped_attrs]

                    train_labels_tr = train_data.iloc[:l, -2:]
                    train_labels_te = train_data.iloc[l:, -2:]

                    model = keras.Sequential()
                    model.add(tf.keras.layers.Dense(dropped_attrs.shape[0], activation='relu',
                                                    kernel_regularizer=keras.regularizers.l2(),
                                                    input_shape=(dropped_attrs.shape[0],)))

                    for width in [dropped_attrs.shape[0]] * 32:  # 32 or 36
                        model.add(
                            tf.keras.layers.Dense(width, activation='relu', kernel_regularizer=keras.regularizers.l2()))

                    model.add(tf.keras.layers.Dense(2))
                    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error',
                                  metrics=['mean_absolute_error'])
                    model.fit(train_features_tr.values, train_labels_tr.values, epochs=35, batch_size=32)
                    rst = model.predict(train_features_te.values)
                    err += calculate_err(train_labels_te.values, rst)

                dropped_attr2err.loc[dropped_attr] = err / n
        if (dropped_attr2err == np.inf).all():
            break
        else:
            optml_attr = dropped_attr2err.idxmin()
            optml_err = dropped_attr2err.min()
            attrs = attrs.drop(optml_attr)

    print(f'sector num = {sector_num}')
    print(f'coverage = {cal_coverage(train_data.loc[:, attrs])}')
    print(f'optimal err = {optml_err}')
    print(attrs)
    pass

    train_features = train_data.loc[:, attrs]
    train_labels = train_data.iloc[:, -2:]

    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(attrs.shape[0], activation='relu', kernel_regularizer=keras.regularizers.l2(),
                                    input_shape=(attrs.shape[0],)))

    for width in [attrs.shape[0]] * 32:  # 32 or 36
        model.add(
            tf.keras.layers.Dense(width, activation='relu', kernel_regularizer=keras.regularizers.l2()))

    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    model.fit(train_features.values, train_labels.values, epochs=35, batch_size=32)
    rst = model.predict(test_features.values)

    rst = pd.DataFrame(rst, columns=['x', 'y'], index=pd.Series(list(range(rst.shape[0])), name='id'))
    rst.to_csv('dnn_new.csv')


def apply_on_dnn():
    attrs = pd.Index(['21000', '21002', '21003', '21005', '21006', '21007', '210012', '35004', '35008', '350014'])

    train_data, test_features = read_data()

    train_features = train_data.loc[:, attrs]
    train_labels = train_data.iloc[:, -2:]

    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(attrs.shape[0], activation='relu', kernel_regularizer=keras.regularizers.l2(),
                                    input_shape=(attrs.shape[0],)))

    for width in [attrs.shape[0]] * 32:  # 32 or 36
        model.add(
            tf.keras.layers.Dense(width, activation='relu', kernel_regularizer=keras.regularizers.l2()))

    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    model.fit(train_features.values, train_labels.values, epochs=60, batch_size=32)
    rst = model.predict(test_features.loc[:, attrs].values)
    rst = pd.DataFrame(rst, columns=['x', 'y'], index=pd.Series(list(range(rst.shape[0])), name='id'))
    rst.to_csv('dnn_new.csv')


def apply_on_knn():
    attrs = pd.Index(['21000', '21002', '21003', '21005', '21006', '21007', '210012', '35004', '35008', '350014'])
    train_data, test_features = read_data()
    train_data = shuffle(train_data)

    l = int(0.8 * train_data.shape[0])

    train_features_tr = train_data.loc[train_data.index[:l], attrs]
    train_features_te = train_data.loc[train_data.index[l:], attrs]

    train_labels_tr = train_data.iloc[:l, -2:]
    train_labels_te = train_data.iloc[l:, -2:]

    rgs = neighbors.KNeighborsRegressor(n_neighbors=2, weights='distance')

    rgs.fit(train_features_tr, train_labels_tr)
    rst = rgs.predict(train_features_te)

    print(calculate_err(rst, train_labels_te))


if __name__ == '__main__':
    main()
