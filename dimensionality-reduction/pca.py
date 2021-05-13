#!/usr/bin/python3
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets._samples_generator import make_blobs


class Pca(object):

    def __init__(self):
        logging.warning('This is PCA algorithm')

    def pca_v1(self):
        # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
        X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
                          cluster_std=[0.2, 0.1, 0.2, 0.2],
                          random_state=9)
        print('------降维前的数据如下------')
        print(X)
        fig = plt.figure()
        ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20, auto_add_to_figure=False)
        fig.add_axes(ax)
        plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
        plt.show()

        # n_components用来指定降维后的特征维度数目
        pca = PCA(n_components=3)
        pca.fit(X)
        print('------降维前的指标如下------')
        print('------explained_variance_ratio_如下------')
        print(pca.explained_variance_ratio_)

        # 开始降维
        # 从3维降到2维
        pca = PCA(n_components=2)
        pca.fit(X)
        print('------降维后的指标如下------')
        # 输出特征值(代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分)
        print('------特征值如下------')
        print(pca.explained_variance_)
        # 输出特征向量
        print('------特征向量如下------')
        print(pca.components_)
        # 输出explained_variance_ratio_
        # (代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分)
        print('------explained_variance_ratio_如下------')
        print(pca.explained_variance_ratio_)
        # 降维后的数据
        X_new = pca.transform(X)
        print('------降维后的数据如下------')
        print(X_new)
        plt.scatter(X_new[:, 0], X_new[:, 1], marker='o')
        plt.show()




if __name__ == '__main__':
    pca_obj = Pca()
    pca_obj.pca_v1()