import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import davies_bouldin_score as dbi
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def pre_process_task2(raw_data=pd.DataFrame()):
    train_data = raw_data.iloc[:, [0, 1, 3]]
    BSSID_col = list(train_data.index.unique())
    idx = train_data.finLabel.unique()
    design_mat = pd.DataFrame(columns=BSSID_col + ['RoomLabel'], index=idx)
    for fin in train_data.finLabel.unique():
        design_mat.loc[fin, :-1] = train_data.iloc[(train_data.loc[:, 'finLabel'] == fin).tolist(), 0]
        design_mat.loc[fin, 'RoomLabel'] = train_data.iloc[(train_data.loc[:, 'finLabel'] == fin).tolist(), 1].iloc[0]
    design_mat = design_mat.fillna(-500)
    return design_mat


def task2():
    raw_data1 = pd.read_csv('./data/Task2/DataSetKMeans1.csv', index_col=0)
    raw_data2 = pd.read_csv('./data/Task2/DataSetKMeans2.csv', index_col=0)
    data1, data2 = pre_process_task2(raw_data1), pre_process_task2(raw_data2)
    data = (data1, data2)
    trans = PCA(n_components=2), PCA(n_components=2)
    # trans = MDS(n_components=2), MDS(n_components=2)
    transformed = trans[0].fit_transform(data1.iloc[:, :-1]), trans[1].fit_transform(data2.iloc[:, :-1])
    dbi_record = pd.DataFrame(columns=pd.Series(list(range(2, 6)) + ['truth'], name='k'),
                              index=pd.Series([1, 2], name='data_idx'))

    for data_idx in (0, 1):
        plt.scatter(transformed[data_idx][:, 0], transformed[data_idx][:, 1],
                    c=data[data_idx].iloc[:, -1], marker='.')
        dbi_score = dbi(data[data_idx].iloc[:, :-1], data[data_idx].iloc[:, -1])
        dbi_record.loc[data_idx + 1, 'truth'] = dbi_score
        plt.title(f'data {data_idx + 1} with true label')
        # plt.savefig(fname=f'data {data_idx + 1} with true label.svg')
        plt.show()
        for k in range(2, 6):
            cls = KMeans(n_clusters=k, init='k-means++')
            predict = cls.fit_predict(data[data_idx].iloc[:, :-1])
            centroids = cls.cluster_centers_
            centroids_trans = trans[data_idx].transform(centroids)
            # centroids, clusterAssment = kMeans(data[data_idx].iloc[:, :-1].values, k)
            # centroids = np.array(centroids)
            dbi_score = dbi(data[data_idx].iloc[:, :-1], predict)
            dbi_record.loc[data_idx + 1, k] = dbi_score
            plt.scatter(transformed[data_idx][:, 0], transformed[data_idx][:, 1], c=predict, marker='.',
                        label='data point', alpha=0.3)
            plt.scatter(centroids_trans[:, 0], centroids_trans[:, 1], marker='+',
                        c=list(range(centroids_trans.shape[0])), label='centroid', edgecolors='black', s=150)
            plt.legend()
            plt.title(f'{k}-means on data {data_idx + 1}')
            # plt.savefig(fname=f'{k}-means on data {data_idx + 1}.svg')
            plt.show()
    print(dbi_record)
    pass


if __name__ == '__main__':
    task2()
