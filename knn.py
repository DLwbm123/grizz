#!/usr/bin/python3
from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import logging

class Knn(object):

    def __init__(self):
        logging.warning('This is KNN algorithm')

    def knn_v1(self):
        X_train, X_test, y_train, y_test = self._get_train_and_test_data()
        # n_neighbors: 就是K
        # metric: 样本之间距离度量公式，默认为闵可夫斯基距离
        # p: 默认值为2，即为欧几里德距离
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X_train, y_train)
        # To caculate the accuracy score
        cor = np.count_nonzero((clf.predict(X_test) == y_test) == True)
        logging.warning('Accuracy is: %s', cor / len(X_test))

    def knn_v2(self):
        k = 3
        X_train, X_test, y_train, y_test = self._get_train_and_test_data()
        predictions = [self.knn_classify(X_train, y_train, data, k) for data in X_test]
        cor = np.count_nonzero((predictions == y_test) == True)
        logging.warning('Accuracy is: %s', cor / len(X_test))

    def get_euclidean_distance(self, instance1, instance2):
        '''
        :param instance1: the first sample(array)
        :param instance2: the second sample(array)
        :return: dist
        '''
        return np.sqrt(sum((instance1 - instance2)**2))

    def knn_classify(self, X, y, testInstance, k):
        '''
        Given testInstance, KNN algorithm is used to predict its label
        :param X: Speciality of training data
        :param y: Label of training data
        :param testInstance:
        :param k: the number of neighbors
        :return:
        '''
        # 计算已知类别数据集中的点与当前点之间的距离
        distances = [self.get_euclidean_distance(x, testInstance) for x in X]
        # 按照距离次序排序并选取与当前点距离最小的k个点
        kneighbors = np.argsort(distances)[:k]
        # 确定前k个点所在类别的出现频率
        count = Counter(y[kneighbors])
        # 返回前k个点出现频率最高的类别作为当前点的预测分类
        return count.most_common()[0][0]

    def _get_train_and_test_data(self):
        # 加载iris数据集
        iris = datasets.load_iris()
        x = iris.data
        y = iris.target
        # 拆分数据集为训练集和测试集
        # random_state代表随机种子编号，保证程序每次运行都分割一样的训练集 & 测试集
        # shuffle代表是否进行有放回抽样
        # test_size表示切割多少百分比的数据做为测试集
        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=2003, shuffle=True)
        return (X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    knn_obj = Knn()
    # knn_obj.knn_v1()
    knn_obj.knn_v2()