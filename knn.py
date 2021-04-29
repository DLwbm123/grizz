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
        # KNeighborsClassifier belongs to sklearn package
        # n_neighbors is K
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
        distances = [self.get_euclidean_distance(x, testInstance) for x in X]
        kneighbors = np.argsort(distances)[:k]
        count = Counter(y[kneighbors])
        return count.most_common()[0][0]

    def _get_train_and_test_data(self):
        # load iris dataset
        iris = datasets.load_iris()
        x = iris.data
        y = iris.target
        # split data into training data & test data
        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=2003)
        return (X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    knn_obj = Knn()
    knn_obj.knn_v1()
    knn_obj.knn_v2()