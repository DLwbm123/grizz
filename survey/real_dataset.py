#!/usr/bin/env python3
import logging

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as tfs
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from representation import GeometricRepresentation

'''
    Just for cifar10 dimensionality reduction
'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 20, 5)
        self.fc1 = nn.Linear(20 * 5 * 5, 200)
        # self.fc2 = nn.Linear(640, 200)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return x


class RealisticDatasets(object):

    def __init__(self):
        logging.info('Obtained realistic datasets')
        self.gr_obj = GeometricRepresentation()


    def get_mnist_data(self):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        return (X, y)

    def get_cifar10_trainloader(self):
        transform = tfs.Compose(
            [tfs.ToTensor(),
             tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 128

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return trainloader

    def get_cifar10_data(self, trainloader):
        net = Net()
        cifar10_output = []
        cifar10_labels = []
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # inputs = self.gr_obj.get_normalized_distribution(inputs, show=False)
            outputs = net(inputs)
            outputs_np_arr = outputs.detach().numpy()
            labels_np_arr = labels.detach().numpy()
            for s_np_arr in outputs_np_arr:
                cifar10_output.append(s_np_arr)
            for label in labels_np_arr:
                cifar10_labels.append(label)
        return (np.array(cifar10_output), np.array(cifar10_labels))

    def get_glass_data(self):
        glass_data = pd.read_csv('./data/glass.data', sep=',', header=None)
        features = glass_data.iloc[:, 1:-1]
        labels = glass_data.iloc[:, -1]
        return (features.to_numpy(), labels.to_numpy())

    def get_spambase_data(self):
        spambase_data = pd.read_csv('./data/spambase.data', sep=',', header=None)
        features = spambase_data.iloc[:, 0:-1]
        labels = spambase_data.iloc[:, -1]
        return (features.to_numpy(), labels.to_numpy())


class DimensionalityReduction(object):

    def pca_dr(self, X, n_components=100):
        pca = PCA(n_components=n_components)
        newX = pca.fit_transform(X)
        return newX


class Mixture(object):

    def get_gaussian_mixture_result(self, input, n_components=1):
        gm = GaussianMixture(n_components=n_components, random_state=0).fit(input)
        _mean = gm.means_
        _cov = gm.covariances_
        return (_mean, _cov)


if __name__ == '__main__':
    rd_obj = RealisticDatasets()
    dr_obj = DimensionalityReduction()
    gr_obj = GeometricRepresentation()
    mixture_obj = Mixture()
    # X, y = rd_obj.get_mnist_data()
    # X = X[0: 60000]
    # y = y[0: 60000]
    # newX = dr_obj.pca_dr(X)
    # #########################euclidean_centroids#############################################
    # plt.scatter(newX[:, 0], newX[:, 1], c='b')
    # mnist_euclidean_centroids, mnist_clusters = gr_obj.get_euclidean_centroids(newX, point_num=10)
    # #########################frechet_means###################################################
    # plt.scatter(newX[:, 0], newX[:, 1], c='b')
    # mnist_frechet_means = gr_obj.get_frechet_means(newX, point_num=10)
    # #########################coreset#########################################################
    # plt.scatter(newX[:, 0], newX[:, 1], c='b')
    # mnist_coreset_idcs = gr_obj.coreset_by_k_greedy(newX)
    # #########################bayesian_coreset################################################
    # mnist_gm_result = mixture_obj.get_gaussian_mixture_result(newX)
    # plt.scatter(newX[:, 0], newX[:, 1], c='b')
    # mnist_bc_coreset = gr_obj.bayesian_coreset_method(newX, distribution_type='mnist', gm_result=mnist_gm_result)
    # #########################poincare_centroids##############################################
    # plt.scatter(newX[:, 0], newX[:, 1], c='b')
    # mnist_poincare_centroids = gr_obj.get_poincare_centroids(newX, point_num=10)
    # #########################lorentzian_centroids############################################
    # plt.scatter(newX[:, 0], newX[:, 1], c='b')
    # mnist_lorentzian_centroids = gr_obj.get_lorentzian_centroids(newX, point_num=10)
    # #########################lorentzian_focal_points#########################################
    # plt.scatter(newX[:, 0], newX[:, 1], c='b')
    # mnist_lorentzian_focal_points = gr_obj.get_lorentzian_focal_points(newX, point_num=10)
    #
    # cifar10_trainloader = rd_obj.get_cifar10_trainloader()
    # cifar10_output, cifar10_labels = rd_obj.get_cifar10_data(cifar10_trainloader)
    # cifar10_centroids, cifar10_clusters = gr_obj.get_euclidean_centroids(cifar10_output[0: 45000], point_num=10)
    # cifar10_frechet_means = gr_obj.get_frechet_means(cifar10_output[0: 45000], point_num=10)
    # cifar10_coreset_idcs = gr_obj.coreset_by_k_greedy(cifar10_output[0: 45000])
    # cifar10_gm_result = mixture_obj.get_gaussian_mixture_result(cifar10_output[0: 45000])
    # cifar10_bc_coreset = gr_obj.bayesian_coreset_method(cifar10_output[0: 45000], distribution_type='cifar10', gm_result=cifar10_gm_result)
    # cifar10_poincare_centroids = gr_obj.get_poincare_centroids(cifar10_output[0: 45000], point_num=10)
    # cifar10_lorentzian_centroids = gr_obj.get_lorentzian_centroids(cifar10_output[0: 45000], point_num=10)
    # cifar10_lorentzian_focal_points = gr_obj.get_lorentzian_focal_points(cifar10_output[0: 45000], point_num=10)
    #
    # glass_data, spambase_labels = rd_obj.get_glass_data()
    # glass_euclidean_centroids = gr_obj.get_euclidean_centroids(glass_data, point_num=10)
    # glass_frechet_means = gr_obj.get_frechet_means(glass_data, point_num=10)
    # glass_coreset_idcs = gr_obj.coreset_by_k_greedy(glass_data)
    # glass_gm_result = mixture_obj.get_gaussian_mixture_result(glass_data)
    # glass_bc_coreset = gr_obj.bayesian_coreset_method(
    #       glass_data, distribution_type='glass', gm_result=glass_gm_result)
    # glass_poincare_centroids = gr_obj.get_poincare_centroids(glass_data, point_num=10)
    # glass_lorentzian_centroids = gr_obj.get_lorentzian_centroids(glass_data, point_num=10)
    # glass_lorentzian_focal_points = gr_obj.get_lorentzian_focal_points(glass_data, point_num=10)

    spambase_data, spambase_labels = rd_obj.get_spambase_data()
    spambase_centroids, spambase_clusters = gr_obj.get_euclidean_centroids(spambase_data[0: 4000], point_num=10)
    spambase_frechet_means = gr_obj.get_frechet_means(spambase_data[0: 4000], point_num=10)
    spambase_coreset_idcs = gr_obj.coreset_by_k_greedy(spambase_data[0: 4000], point_num=10)
    spambase_gm_result = mixture_obj.get_gaussian_mixture_result(spambase_data[0: 4000])
    spambase_bc_coreset = gr_obj.bayesian_coreset_method(spambase_data[0: 4000], distribution_type='spambase', gm_result=spambase_gm_result, point_num=100, fill=True)
    spambase_poincare_centroids, spambase_poincare_clusters = gr_obj.get_poincare_centroids(spambase_data[0: 4000], point_num=10)
    spambase_lorentzian_centroids = gr_obj.get_lorentzian_centroids(spambase_data[0: 4000], point_num=10)
    spambase_lorentzian_focal_points = gr_obj.get_lorentzian_focal_points(spambase_data[0: 4000], point_num=10)