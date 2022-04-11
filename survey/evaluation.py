#!/usr/bin/env python3
import logging
import json
import scipy.stats
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
from common_coreset import kCenterGreedy
from gaussian import GaussianDistribution
from real_dataset import RealisticDatasets, DimensionalityReduction, Mixture
from representation import GeometricRepresentation, DistanceMetrics
from utils import get_cfg_data, get_labels
from sklearn import svm
import matplotlib.pyplot as plt


class KLDivergence(object):
    def __init__(self, kernel='gaussian', bandwidth=0.8):
        self.kernel = kernel
        self.bandwidth = bandwidth
        logging.info(
                'Caculate KL divergence between the subset selected by our algorithm and the whole dataset!')

    def get_KL_divergence(self, distribution, subset_distribution):
        kd_obj = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        kde_all = kd_obj.fit(distribution)
        score_all = np.exp(kde_all.score_samples(distribution))
        kde_subset = kd_obj.fit(subset_distribution)
        score_all_v2 = np.exp(kde_subset.score_samples(distribution))
        KL_divergence = scipy.stats.entropy(score_all, score_all_v2)
        # KL_divergence = scipy.special.kl_div(distribution, subset_distribution)
        return KL_divergence


class MaximumMeanDiscrepancy(nn.Module):

    def __init__(self, kernel_mul=2.0, kernel_num=5):
        logging.info(
                'Caculate Maximum Mean Discrepancy between the subset selected and the whole dataset!')
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        super(MaximumMeanDiscrepancy, self).__init__()

    def get_guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        :param source: the source data-points set(whole-set)
        :param target: the target data-points set(subset)
        :param kernel_mul: to caculate the bandwidth of each kernel function
        :param kernel_num: the number of kernels
        :param fix_sigma: whether to apply fixed sigma
        :return:
                [   K_ss K_st
                    K_ts K_tt ]
        '''
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        # total0 - total1 计算每一个点之间的距离
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2) # 计算高斯核中的分子部分
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            # 计算K(1/(m(m-1)))
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个值
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        # 不同的高斯核
        kernel_val = [torch.exp(-L2_distance / bandwidth_t) for bandwidth_t in bandwidth_list]
        return sum(kernel_val)

    def forward_v2(self, source, target):
        source_size = int(source.size()[0])
        target_size = int(target.size()[0])
        kernels = self.get_guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                  fix_sigma=self.fix_sigma)
        XX = kernels[:source_size, :source_size]
        YY = kernels[source_size:, source_size:]
        XY = kernels[:source_size, source_size:]
        YX = kernels[source_size:, :source_size]

        XX = torch.div(XX, source_size * source_size).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
        XY = torch.div(XY, -source_size * target_size).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

        YX = torch.div(YX, -target_size * source_size).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
        YY = torch.div(YY, target_size * target_size).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

        loss = (XX + XY).sum() + (YX + YY).sum()
        loss = torch.mean(XX + YY - XY - YX)
        return loss

    def mmd_rbf(self, X, Y, gamma=1.0):
        """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            gamma {float} -- [kernel parameter] (default: {1.0})
        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.rbf_kernel(X, X, gamma)
        YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
        XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def forward(self, source, target):
        return self.mmd_rbf(source, target)


class SVMClassifier(object):

    def __init__(self, C, kernel='linear'):
        self.C = C
        self.kernel = kernel
        self.clf = svm.SVC(C=self.C, kernel=self.kernel)
        self.rd_obj = RealisticDatasets()
        self.dr_obj = DimensionalityReduction()
        self.gr_obj = GeometricRepresentation()
        cfg = get_cfg_data()
        self.test_gd_obj = GaussianDistribution(cfg['space-version']['type'], cfg['extra-guassian-params']['items'])
        logging.info(
                'Realize SVMClassifier for comparing the performance of the subset selected and the whole dataset!')

    def get_prepared_data(
            self, distribution, distribution_type, subset, gd_obj, original_data, distance_func, subset_idx=[], need_normalization=False):
        if distribution_type == 'guassian':
            gd_size_list = gd_obj.get_size_list()
            gd_labels = get_labels(gd_size_list)
            if not subset_idx:
                subset_labels = [gd_labels[idx]
                                 for idx in self.get_nearest_neighbor_indices(distance_func, distribution, subset)]
            else:
                subset = distribution[subset_idx]
                subset_labels = [gd_labels[idx] for idx in subset_idx]
            test_gd_distribution = self.test_gd_obj.get_guassian_distribution()
            if need_normalization:
                test_gd_distribution = self.gr_obj.get_normalized_distribution(test_gd_distribution, show=False)
            test_gd_size_list = self.test_gd_obj.get_size_list()
            test_labels = get_labels(test_gd_size_list)
            return (
                distribution, gd_labels, subset,
                subset_labels, test_gd_distribution, test_labels
            )
        elif distribution_type == 'mnist':
            newX = original_data[0]
            newY = original_data[1] # labels
            if need_normalization:
                test_distribution = self.gr_obj.get_normalized_distribution(newX[60000: 70000], show=False)
            else:
                test_distribution = newX[60000: 70000]
            if not subset_idx:
                neighbor_indices = self.get_nearest_neighbor_indices(distance_func, distribution, subset)
                subset_labels = [newY[idx[0]] for idx in neighbor_indices]
            else:
                subset_labels = [newY[idx] for idx in subset_idx]
            test_labels = newY[60000: 70000]
            return (
                distribution, newY[0: 60000], subset,
                subset_labels, test_distribution, test_labels
            )
        elif distribution_type == 'cifar10':
            cifar10_data = original_data[0]
            cifar10_labels = original_data[1]
            if need_normalization:
                test_distribution = self.gr_obj.get_normalized_distribution(cifar10_data[45000: 50000], show=False)
            else:
                test_distribution = cifar10_data[45000: 50000]
            if not subset_idx:
                neighbor_indices = self.get_nearest_neighbor_indices(distance_func, distribution, subset)
                subset_labels = [cifar10_labels[idx[0]]
                                 for idx in neighbor_indices]
            else:
                subset_labels = [cifar10_labels[idx] for idx in subset_idx]
            test_labels = cifar10_labels[45000: 50000]
            return (
                distribution, cifar10_labels[0: 45000], subset,
                subset_labels, test_distribution, test_labels
            )
        elif distribution_type == 'spambase':
            spambase_data = original_data[0]
            spambase_labels = original_data[1]
            if need_normalization:
                test_distribution = self.gr_obj.get_normalized_distribution(spambase_data[4000: 4601], show=False)
            else:
                # the data normalized before being used
                test_distribution = spambase_data[4000: 4601]
            if not subset_idx:
                neighbor_indices = self.get_nearest_neighbor_indices(distance_func, distribution, subset)
                subset_labels = [spambase_labels[idx[0]]
                                 for idx in neighbor_indices]
            else:
                subset_labels = [spambase_labels[idx] for idx in subset_idx]
            test_labels = spambase_labels[4000: 4601]
            return (
                distribution, spambase_labels[0: 4000], subset,
                subset_labels, test_distribution, test_labels
            )

    def get_nearest_neighbor_indices(self, metric_function, distribution, subset):
        # pyfunc = DistanceMetric.get_metric('pyfunc', func=metric_function)
        # dist = pyfunc.pairwise(subset, distribution)
        nbrs = NearestNeighbors(n_neighbors=1, metric=metric_function, algorithm='ball_tree').fit(distribution)
        distances, indices = nbrs.kneighbors(subset)
        return indices

    def train(self, prepared_data):
        final_accuracy_result = {}
        x_train, y_train, cx_train, cy_train, x_test, y_test = prepared_data
        print('--------------------------------------------------------------------------------')
        self.clf.fit(x_train, np.array(y_train))
        print('--------------------------------------------------------------------------------')
        full_training_data_acc = self.clf.score(x_train, y_train)
        full_testing_data_acc = self.clf.score(x_test, y_test)
        final_accuracy_result['full_training_data_acc'] = full_training_data_acc
        final_accuracy_result['full_testing_data_acc'] = full_testing_data_acc
        print("training data accuracy: ", full_training_data_acc)
        print("testing data accuracy: ", full_testing_data_acc)
        print('--------------------------------------------------------------------------------')
        self.clf.fit(cx_train, np.array(cy_train))
        print('--------------------------------------------------------------------------------')
        subset_training_data_acc = self.clf.score(cx_train, cy_train)
        subset_testing_data_acc = self.clf.score(x_test, y_test)
        final_accuracy_result['subset_training_data_acc'] = subset_training_data_acc
        final_accuracy_result['subset_testing_data_acc'] = subset_testing_data_acc
        print("coreset training data accuracy: ", subset_training_data_acc)
        print("coreset testing data accuracy: ", subset_testing_data_acc)
        return final_accuracy_result


class ResultDisplay(object):

    def __init__(self):
        cfg = get_cfg_data()
        self.gd_obj = GaussianDistribution(cfg['space-version']['type'], cfg['guassian-params']['items'])
        self.gd_distribution = self.gd_obj.get_guassian_distribution()
        self.gr_obj = GeometricRepresentation()
        self.kcg_obj = kCenterGreedy(self.gd_distribution)
        self.kld_obj = KLDivergence()
        self.mmd_obj = MaximumMeanDiscrepancy()
        self.dm_obj = DistanceMetrics()
        self.svm_obj = SVMClassifier(10, 'rbf')
        self.color_options = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'pink']

    def kl_divergence(self, distribution, distribution_type, gm_result=None):
        kl_divergence_map = {}
        self.kcg_obj = kCenterGreedy(distribution)
        if distribution_type == 'guassian':
            for n in range(50, 600, 50):
                self.do_kl_divergence(n, distribution, distribution_type, kl_divergence_map, gm_result)
        elif distribution_type in ('mnist', 'cifar10', 'spambase'):
            for n in range(50, 600, 50):
                self.do_kl_divergence(n, distribution, distribution_type, kl_divergence_map, gm_result)
        title = 'KL Divergence with different batch settings'
        y_label = 'KL Divergence'
        dataset_type = 'kl_divergence_{}'.format(distribution_type)
        self.write_result_to_file(kl_divergence_map, dataset_type)
        self.plot_result(title, kl_divergence_map, y_label)

    def do_kl_divergence(self, n, distribution, distribution_type, kl_divergence_map, gm_result):
        # Euclidean centroids
        if distribution_type == 'guassian':
            self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        euclidean_centroids, euclidean_clusters = self.gr_obj.get_euclidean_centroids(distribution, point_num=n)
        kl_divergence_map.setdefault('euclidean centroids', [])
        euclidean_centroids_kl_divergence = self.kld_obj.get_KL_divergence(distribution, euclidean_centroids)
        print('--------------------------------------------------------------------------------')
        print('KL Divergence: {} with n: {} for euclidean centroids'.format(euclidean_centroids_kl_divergence, n))
        kl_divergence_map['euclidean centroids'].append((n, euclidean_centroids_kl_divergence))

        # Frechet means
        if distribution_type == 'guassian':
            self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        frechet_means = self.gr_obj.get_frechet_means(distribution, point_num=n)
        kl_divergence_map.setdefault(u'fr ́echet means', [])
        frechet_means_kl_divergence = self.kld_obj.get_KL_divergence(distribution, frechet_means)
        print('--------------------------------------------------------------------------------')
        print('KL Divergence: {} with n: {} for frechet means'.format(frechet_means_kl_divergence, n))
        kl_divergence_map[u'fr ́echet means'].append((n, frechet_means_kl_divergence))

        # Lorentzian centroids
        if distribution_type == 'guassian':
            self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        lorentzian_centroids, clusters = self.gr_obj.get_lorentzian_centroids(distribution, point_num=n)
        kl_divergence_map.setdefault('lorentzian centroids', [])
        lorentzian_centroids_kl_divergence = self.kld_obj.get_KL_divergence(distribution, lorentzian_centroids)
        print('--------------------------------------------------------------------------------')
        print('KL Divergence: {} with n: {} for lorentzian centroids'.format(lorentzian_centroids_kl_divergence, n))
        kl_divergence_map['lorentzian centroids'].append((n, lorentzian_centroids_kl_divergence))

        # Lorentzian focal points
        if distribution_type == 'guassian':
            self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        lorentzian_focal_points = self.gr_obj.get_lorentzian_focal_points(distribution, point_num=n)
        kl_divergence_map.setdefault('lorentzian focal points', [])
        lorentzian_focal_points_kl_divergence = self.kld_obj.get_KL_divergence(distribution, lorentzian_focal_points)
        print('--------------------------------------------------------------------------------')
        print(
            'KL Divergence: {} with n: {} for lorentzian focal points'.format(lorentzian_focal_points_kl_divergence, n))
        kl_divergence_map['lorentzian focal points'].append((n, lorentzian_focal_points_kl_divergence))

        # # KL Divergence for coreset
        # kl_divergence_map.setdefault('coreset', [])
        # idx_list = self.kcg_obj.select_batch_(model=None, already_selected=[], N=n)
        # coreset_kl_divergence = self.kld_obj.get_KL_divergence(distribution, distribution[idx_list])
        # print('--------------------------------------------------------------------------------')
        # print('KL Divergence: {} with n: {} for common coreset'.format(coreset_kl_divergence, n))
        # kl_divergence_map['coreset'].append((n, coreset_kl_divergence))

        # KL Divergence for Bayesian coreset
        # kl_divergence_map.setdefault('bayesian coreset', [])
        # if distribution_type == 'guassian':
        #     self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        # bayesian_coreset = self.gr_obj.bayesian_coreset_method(distribution, distribution_type=distribution_type,
        #                                                        gm_result=gm_result, point_num=n)
        # bc_kl_divergence = self.kld_obj.get_KL_divergence(distribution, bayesian_coreset)
        # print('--------------------------------------------------------------------------------')
        # print('KL Divergence: {} with n: {} for bayesian coreset'.format(bc_kl_divergence, n))
        # kl_divergence_map['bayesian coreset'].append((n, bc_kl_divergence))

        # KL Divergence for Bayesian coreset++
        # kl_divergence_map.setdefault('bayesian coreset++', [])
        # if distribution_type == 'guassian':
        #     self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        # bayesian_coreset_plus = self.gr_obj.bayesian_coreset_method(distribution, distribution_type=distribution_type,
        #                                                        gm_result=gm_result, point_num=n, fill=True)
        # bc_plus_kl_divergence = self.kld_obj.get_KL_divergence(distribution, bayesian_coreset_plus)
        # print('--------------------------------------------------------------------------------')
        # print('KL Divergence: {} with n: {} for bayesian coreset++'.format(bc_plus_kl_divergence, n))
        # kl_divergence_map['bayesian coreset++'].append((n, bc_plus_kl_divergence))

        # Poincare centroids
        normalized_distribution = distribution
        if distribution_type == 'guassian':
            normalized_distribution = self.gr_obj.get_normalized_distribution(distribution, show=False)
            self.gd_obj.display_gd_distribution(normalized_distribution, show=False, scatt=True)
        poincare_centroids, poincare_clusters = self.gr_obj.get_poincare_centroids(normalized_distribution, point_num=n)
        kl_divergence_map.setdefault(u'poincar ́e centroids', [])
        poincare_centroids_kl_divergence = self.kld_obj.get_KL_divergence(normalized_distribution, poincare_centroids)
        print('--------------------------------------------------------------------------------')
        print('KL Divergence: {} with n: {} for poincare centroids'.format(poincare_centroids_kl_divergence, n))
        kl_divergence_map[u'poincar ́e centroids'].append((n, poincare_centroids_kl_divergence))

    def mmd(self, distribution, distribution_type, gm_result=None):
        mmd_map = {}
        self.kcg_obj = kCenterGreedy(distribution)
        if distribution_type == 'guassian':
            for n in range(50, 600, 50):
                self.do_mmd(n, distribution, distribution_type, mmd_map, gm_result)
        elif distribution_type in ('mnist', 'cifar10', 'spambase'):
            for n in range(50, 600, 50):
                self.do_mmd(n, distribution, distribution_type, mmd_map, gm_result)
        title = 'Maximum Mean Discrepancy with different batch settings'
        y_label = 'Maximum Mean Discrepancy'
        dataset_type = 'mmd_{}'.format(distribution_type)
        self.write_result_to_file(mmd_map, dataset_type)
        self.plot_result(title, mmd_map, y_label)

    def do_mmd(self, n, distribution, distribution_type, mmd_map, gm_result):
        # Euclidean centroids
        if distribution_type == 'guassian':
            self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        euclidean_centroids, euclidean_clusters = self.gr_obj.get_euclidean_centroids(distribution, point_num=n)
        mmd_map.setdefault('euclidean centroids', [])
        euclidean_centroids_mmd = self.mmd_obj.forward(torch.tensor(distribution), torch.tensor(euclidean_centroids))
        print('--------------------------------------------------------------------------------')
        print('Maximum Mean Discrepancy: {} with n: {} for euclidean centroids'.format(euclidean_centroids_mmd.item(), n))
        mmd_map['euclidean centroids'].append((n, euclidean_centroids_mmd.item()))

        # Frechet means
        if distribution_type == 'guassian':
            self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        frechet_means = self.gr_obj.get_frechet_means(distribution, point_num=n)
        mmd_map.setdefault(u'fr ́echet means', [])
        frechet_means_mmd = self.mmd_obj.forward(distribution, frechet_means)
        print('--------------------------------------------------------------------------------')
        print('Maximum Mean Discrepancy: {} with n: {} for frechet means'.format(frechet_means_mmd.item(), n))
        mmd_map[u'fr ́echet means'].append((n, frechet_means_mmd.item()))

        # Lorentzian centroids
        if distribution_type == 'guassian':
            self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        lorentzian_centroids, clusters = self.gr_obj.get_lorentzian_centroids(distribution, point_num=n)
        mmd_map.setdefault('lorentzian centroids', [])
        lorentzian_centroids_mmd = self.mmd_obj.forward(distribution, lorentzian_centroids)
        print('--------------------------------------------------------------------------------')
        print('Maximum Mean Discrepancy: {} with n: {} for lorentzian centroids'.format(lorentzian_centroids_mmd.item(),
                                                                                        n))
        mmd_map['lorentzian centroids'].append((n, lorentzian_centroids_mmd.item()))

        # Lorentzian focal points
        if distribution_type == 'guassian':
            self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        lorentzian_focal_points = self.gr_obj.get_lorentzian_focal_points(distribution, point_num=n)
        mmd_map.setdefault('lorentzian focal points', [])
        lorentzian_focal_points_mmd = self.mmd_obj.forward(distribution, lorentzian_focal_points)
        print('--------------------------------------------------------------------------------')
        print('Maximum Mean Discrepancy: {} with n: {} for lorentzian focal points'.format(
            lorentzian_focal_points_mmd.item(), n))
        mmd_map['lorentzian focal points'].append((n, lorentzian_focal_points_mmd.item()))

        # Coreset
        # mmd_map.setdefault('coreset', [])
        # idx_list = self.kcg_obj.select_batch_(model=None, already_selected=[], N=n)
        # coreset_mmd = self.mmd_obj.forward(distribution, distribution[idx_list])
        # print('--------------------------------------------------------------------------------')
        # print('Maximum Mean Discrepancy: {} with n: {} for common coreset'.format(coreset_mmd.item(), n))
        # mmd_map['coreset'].append((n, coreset_mmd.item()))

        # Bayesian coreset
        # mmd_map.setdefault('bayesian coreset', [])
        # if distribution_type == 'guassian':
        #     self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        # bayesian_coreset = self.gr_obj.bayesian_coreset_method(distribution, distribution_type=distribution_type,
        #                                                        gm_result=gm_result, point_num=n)
        # bc_mmd = self.mmd_obj.forward(distribution, bayesian_coreset)
        # print('--------------------------------------------------------------------------------')
        # print('Maximum Mean Discrepancy: {} with n: {} for bayesian coreset'.format(bc_mmd.item(), n))
        # mmd_map['bayesian coreset'].append((n, bc_mmd.item()))

        # Bayesian coreset++
        # mmd_map.setdefault('bayesian coreset++', [])
        # if distribution_type == 'guassian':
        #     self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        # bayesian_coreset_plus = self.gr_obj.bayesian_coreset_method(distribution, distribution_type=distribution_type,
        #                                                        gm_result=gm_result, point_num=n, fill=True)
        # bc_plus_mmd = self.mmd_obj.forward(distribution, bayesian_coreset_plus)
        # print('--------------------------------------------------------------------------------')
        # print('Maximum Mean Discrepancy: {} with n: {} for bayesian coreset++'.format(bc_plus_mmd.item(), n))
        # mmd_map['bayesian coreset++'].append((n, bc_plus_mmd.item()))

        # Poincare centroids
        normalized_distribution = distribution
        if distribution_type == 'guassian':
            normalized_distribution = self.gr_obj.get_normalized_distribution(distribution, show=False)
            self.gd_obj.display_gd_distribution(normalized_distribution, show=False, scatt=True)
        poincare_centroids, poincare_clusters = self.gr_obj.get_poincare_centroids(normalized_distribution, point_num=n)
        mmd_map.setdefault(u'poincar ́e centroids', [])
        poincare_centroids_mmd = self.mmd_obj.forward(normalized_distribution, poincare_centroids)
        print('--------------------------------------------------------------------------------')
        print('Maximum Mean Discrepancy: {} with n: {} for poincare centroids'.format(poincare_centroids_mmd.item(), n))
        mmd_map[u'poincar ́e centroids'].append((n, poincare_centroids_mmd.item()))

    def svm(self, distribution, distribution_type, original_data=None, gm_result=None):
        svm_map = {}
        self.kcg_obj = kCenterGreedy(distribution)
        if distribution_type == 'guassian':
            for n in range(50, 600, 50):
                self.do_svm(n, distribution, distribution_type, svm_map, original_data, gm_result)
        elif distribution_type in ('mnist', 'cifar10', 'spambase'):
            for n in range(50, 600, 50):
                self.do_svm(n, distribution, distribution_type, svm_map, original_data, gm_result)
        title = 'SVM testing accuracy with different batch settings'
        y_label = 'SVM testing accuracy'
        dataset_type = 'svm_{}'.format(distribution_type)
        self.write_result_to_file(svm_map, dataset_type)
        self.plot_result(title, svm_map, y_label)

    def do_svm(self, n, distribution, distribution_type, svm_map, original_data, gm_result):
        # Euclidean centroids
        if distribution_type == 'guassian':
            self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        euclidean_centroids, euclidean_clusters = self.gr_obj.get_euclidean_centroids(distribution, point_num=n)
        svm_map.setdefault('euclidean centroids', [])
        prepared_data = self.svm_obj.get_prepared_data(
            distribution, distribution_type, euclidean_centroids, self.gd_obj, original_data,
            self.dm_obj.euclidean_distance_func())
        final_accuracy_result = self.svm_obj.train(prepared_data)
        print('--------------------------------------------------------------------------------')
        print('SVM subset testing accuracy: {} with n: {} for euclidean centroids'.format(
            final_accuracy_result['subset_testing_data_acc'], n))
        svm_map['euclidean centroids'].append((n, final_accuracy_result['subset_testing_data_acc']))

        # Frechet means
        if distribution_type == 'guassian':
            self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        frechet_means = self.gr_obj.get_frechet_means(distribution, point_num=n)
        svm_map.setdefault(u'fr ́echet means', [])
        prepared_data = self.svm_obj.get_prepared_data(
            distribution, distribution_type, frechet_means, self.gd_obj, original_data,
            self.dm_obj.frechet_mean_distance_func())
        final_accuracy_result = self.svm_obj.train(prepared_data)
        print('--------------------------------------------------------------------------------')
        print('SVM subset testing accuracy: {} with n: {} for frechet means'.format(
            final_accuracy_result['subset_testing_data_acc'], n))
        svm_map[u'fr ́echet means'].append((n, final_accuracy_result['subset_testing_data_acc']))

        # Lorentzian centroids
        if distribution_type == 'guassian':
            self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        lorentzian_centroids, clusters = self.gr_obj.get_lorentzian_centroids(distribution, point_num=n)
        svm_map.setdefault('lorentzian centroids', [])
        prepared_data = self.svm_obj.get_prepared_data(
            distribution, distribution_type, lorentzian_centroids, self.gd_obj, original_data,
            self.dm_obj.lorentzian_distance_func())
        final_accuracy_result = self.svm_obj.train(prepared_data)
        print('--------------------------------------------------------------------------------')
        print('SVM subset testing accuracy: {} with n: {} for lorentzian centroids'.format(
            final_accuracy_result['subset_testing_data_acc'], n))
        svm_map['lorentzian centroids'].append((n, final_accuracy_result['subset_testing_data_acc']))

        # Lorentzian focal points
        if distribution_type == 'guassian':
            self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        lorentzian_focal_points = self.gr_obj.get_lorentzian_focal_points(distribution, point_num=n)
        svm_map.setdefault('lorentzian focal points', [])
        prepared_data = self.svm_obj.get_prepared_data(
            distribution, distribution_type, lorentzian_focal_points, self.gd_obj, original_data,
            self.dm_obj.lorentzian_distance_func())
        final_accuracy_result = self.svm_obj.train(prepared_data)
        print('--------------------------------------------------------------------------------')
        print('SVM subset testing accuracy: {} with n: {} for lorentzian focal points'.format(
            final_accuracy_result['subset_testing_data_acc'], n))
        svm_map['lorentzian focal points'].append((n, final_accuracy_result['subset_testing_data_acc']))

        # Coreset
        # svm_map.setdefault('coreset', [])
        # idx_list = self.kcg_obj.select_batch_(model=None, already_selected=[], N=n)
        # subset = distribution[idx_list]
        # prepared_data = self.svm_obj.get_prepared_data(
        #     distribution, distribution_type, subset, self.gd_obj, original_data, self.dm_obj.euclidean_distance_func(),
        #     subset_idx=idx_list)
        # final_accuracy_result = self.svm_obj.train(prepared_data)
        # print('--------------------------------------------------------------------------------')
        # print('SVM subset testing accuracy: {} with n: {} for coreset'.format(
        #     final_accuracy_result['subset_testing_data_acc'], n))
        # svm_map['coreset'].append((n, final_accuracy_result['subset_testing_data_acc']))

        # Bayesian coreset
        # svm_map.setdefault('bayesian coreset', [])
        # if distribution_type == 'guassian':
        #     self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        # bayesian_coreset = self.gr_obj.bayesian_coreset_method(distribution, distribution_type=distribution_type,
        #                                                        gm_result=gm_result, point_num=n)
        # prepared_data = self.svm_obj.get_prepared_data(
        #     distribution, distribution_type, bayesian_coreset, self.gd_obj, original_data,
        #     self.dm_obj.euclidean_distance_func())
        # final_accuracy_result = self.svm_obj.train(prepared_data)
        # print('--------------------------------------------------------------------------------')
        # print('SVM subset testing accuracy: {} with n: {} for bayesian coreset'.format(
        #     final_accuracy_result['subset_testing_data_acc'], n))
        # svm_map['bayesian coreset'].append((n, final_accuracy_result['subset_testing_data_acc']))

        # Bayesian coreset++
        # svm_map.setdefault('bayesian coreset++', [])
        # if distribution_type == 'guassian':
        #     self.gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
        # bayesian_coreset_plus = self.gr_obj.bayesian_coreset_method(distribution, distribution_type=distribution_type,
        #                                                        gm_result=gm_result, point_num=n, fill=True)
        # prepared_data = self.svm_obj.get_prepared_data(
        #         distribution, distribution_type, bayesian_coreset_plus, self.gd_obj, original_data, self.dm_obj.euclidean_distance_func())
        # final_accuracy_result = self.svm_obj.train(prepared_data)
        # print('--------------------------------------------------------------------------------')
        # print('SVM subset testing accuracy: {} with n: {} for bayesian coreset'.format(final_accuracy_result['subset_testing_data_acc'], n))
        # svm_map['bayesian coreset++'].append((n, final_accuracy_result['subset_testing_data_acc']))

        need_normalization = False
        # Poincare centroids
        normalized_distribution = distribution
        if distribution_type == 'guassian':
            need_normalization = True
            normalized_distribution = self.gr_obj.get_normalized_distribution(distribution, show=False)
            self.gd_obj.display_gd_distribution(normalized_distribution, show=False, scatt=True)
        poincare_centroids, poincare_clusters = self.gr_obj.get_poincare_centroids(normalized_distribution, point_num=n)
        svm_map.setdefault(u'poincar ́e centroids', [])
        prepared_data = self.svm_obj.get_prepared_data(normalized_distribution, distribution_type, poincare_centroids,
                                                       self.gd_obj, original_data,
                                                       self.dm_obj.poincare_centroid_distance_func(),
                                                       need_normalization=need_normalization)
        final_accuracy_result = self.svm_obj.train(prepared_data)
        print('--------------------------------------------------------------------------------')
        print('SVM subset testing accuracy: {} with n: {} for poincare centroids'.format(
            final_accuracy_result['subset_testing_data_acc'], n))
        svm_map[u'poincar ́e centroids'].append((n, final_accuracy_result['subset_testing_data_acc']))

    def plot_result(self, title, data_map, y_label):
        plt.figure(title, figsize=(10, 6))
        t = 0
        for name, kl_data in data_map.items():
            x_list = []
            y_list = []
            for da in kl_data:
                x_list.append(da[0])
                y_list.append(da[1])
            plt.plot(x_list, y_list, color=self.color_options[t], marker='o', linestyle='-.', label=name)
            t += 1
        plt.legend()
        plt.xlabel('subset size')
        plt.ylabel(y_label)
        figure_name = '{}.eps'.format(title)
        plt.savefig(figure_name, bbox_inches='tight', dpi=600, format='eps')
        plt.show()

    def write_result_to_file(self, result, dataset_type):
        file_name = '{}.txt'.format(dataset_type)
        with open(file_name, 'w') as convert_file:
            convert_file.write(json.dumps(result))


if __name__ == '__main__':
    rd = ResultDisplay()
    rd_obj = RealisticDatasets()
    dr_obj = DimensionalityReduction()
    gr_obj = GeometricRepresentation()
    mixture_obj = Mixture()

    # print('Evaluation for Guassian distribution')
    # rd.kl_divergence(rd.gd_distribution, 'guassian')
    # rd.mmd(rd.gd_distribution, 'guassian')
    # rd.svm(rd.gd_distribution, 'guassian')

    # print('Evaluation for spambase distribution')
    # spambase_data, spambase_labels = rd_obj.get_spambase_data()
    # normalized_spambase_data = gr_obj.get_normalized_distribution(spambase_data, show=False)
    # spambase_gm_result = mixture_obj.get_gaussian_mixture_result(normalized_spambase_data[0: 4000])
    # rd.kl_divergence(normalized_spambase_data[0: 4000], 'spambase', gm_result=spambase_gm_result)
    # rd.mmd(normalized_spambase_data[0: 4000], 'spambase', gm_result=spambase_gm_result)
    # rd.svm(normalized_spambase_data[0: 4000], 'spambase', original_data=(normalized_spambase_data, spambase_labels), gm_result=spambase_gm_result)

    #print('Evaluation for MNIST distribution')
    #X, y = rd_obj.get_mnist_data()
    #normalized_X = gr_obj.get_normalized_distribution(X, show=False)
    #newX = dr_obj.pca_dr(normalized_X)
    #newY = [int(val) for _, val in y.iteritems()]
    #mnist_gm_result = mixture_obj.get_gaussian_mixture_result(newX[0: 60000])
    #rd.kl_divergence(newX[0: 60000], 'mnist', gm_result=mnist_gm_result)
    #rd.mmd(newX[0: 60000], 'mnist', gm_result=mnist_gm_result)
    #rd.svm(newX[0: 60000], 'mnist', original_data=(newX, newY), gm_result=mnist_gm_result)

    print('Evaluation for Cifar10 distribution')
    cifar10_trainloader = rd_obj.get_cifar10_trainloader()
    # dimensional reduction realized in inner part of `get_cifar10_data`
    cifar10_output, cifar10_labels = rd_obj.get_cifar10_data(cifar10_trainloader)
    cifar10_gm_result = mixture_obj.get_gaussian_mixture_result(cifar10_output[0: 45000])
    rd.kl_divergence(cifar10_output[0: 45000], 'cifar10', gm_result=cifar10_gm_result)
    #rd.mmd(cifar10_output[0: 45000], 'cifar10', gm_result=cifar10_gm_result)
    #rd.svm(cifar10_output[0: 45000], 'cifar10', original_data=(cifar10_output, cifar10_labels), gm_result=cifar10_gm_result)
