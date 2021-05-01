#!/usr/bin/python3
import numpy
from sklearn import metrics
import numpy as np
import logging


class Mds(object):

    def __init__(self):
        logging.warning('This is MDS algorithm')

    def get_euclidean_distance(self, instance1, instance2):
        '''
        :param instance1: the first sample(array)
        :param instance2: the second sample(array)
        :return: dist
        '''
        return np.sqrt(sum((instance1 - instance2) ** 2))

    # 计算矩阵各行之间的欧式距离；
    # x矩阵的第i行与y矩阵的第0-j行继续欧式距离计算，构成新矩阵第i行[i0、i1...ij]
    def calculate_distance_matrix(self, x, y):
        # 两个矩阵样本之间的距离(成对距离)
        d = metrics.pairwise_distances(x, y)
        print('------原始距离矩阵如下------')
        print(d)
        return d

    def cal_B(self, D):
        (n1, n2) = D.shape
        DD = numpy.square(D)  # 矩阵D 所有元素平方
        Di = numpy.sum(DD, axis=1) / n1  # 计算dist(i.)^2
        Dj = numpy.sum(DD, axis=0) / n1  # 计算dist(.j)^2
        Dij = numpy.sum(DD) / (n1 ** 2)  # 计算dist(ij)^2
        B = numpy.zeros((n1, n1))
        for i in range(n1):
            for j in range(n2):
                # 利用公式求bij
                B[i, j] = (Dij + DD[i, j] - Di[i] - Dj[j]) / (-2)  # 计算b(ij)
        return B

    def mds_v1(self, data, n=2):
        # 计算原始空间中数据点的距离矩阵
        D = self.calculate_distance_matrix(data, data)
        # 计算内积矩阵B
        B = self.cal_B(D)
        # 对B进行特征分解
        # Be为矩阵B的特征值，Bv为对应的的特征向量
        Be, Bv = numpy.linalg.eigh(B)
        # 特征值从大到小排序
        Be_sort = numpy.argsort(-Be)
        Be = Be[Be_sort]
        # 特征值对应的特征向量
        Bv = Bv[:, Be_sort]

        # 前n个特征值对角矩阵
        Bez = numpy.diag(Be[0:n])
        # 前n个特征向量
        Bvz = Bv[:, 0:n]
        Z = numpy.dot(numpy.sqrt(Bez), Bvz.T).T
        return Z


if __name__ == '__main__':
    mds_obj = Mds()
    data = numpy.mat([[3, 2, 4], [2, 0, 2], [4, 2, 4]])
    final_result = mds_obj.mds_v1(data)
    print('------降维后矩阵如下------')
    print(final_result)