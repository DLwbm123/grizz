#!/usr/bin/env python3
import gc
import logging
import random

import numpy as np
import torch
import math
import sklearn
from abc import abstractmethod
from bayesian_coreset import BayesianCoreset
from gaussian import GaussianDistribution
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from common_coreset import kCenterGreedy
from manifold import PoincareManifold, LorentzManifold
from utils import get_cfg_data
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric, euclidean_distance

class DistanceMetrics(object):

    def __init__(self):
        logging.info('Get different distance metrics for tasks')

    def euclidean_distance_func(self):
        return lambda x, y: euclidean_distance(x, y)

    def frechet_mean_distance_func(self):
        return lambda x, y: self.get_frechet_mean_distance(x, y)

    def get_frechet_mean_distance(self, u, v):
        ed = euclidean_distance(u, v)
        return ed ** 2

    def poincare_centroid_distance_func(self):
        poincare_mf_obj = PoincareManifold()
        return lambda x, y: poincare_mf_obj.distance(torch.from_numpy(x), torch.from_numpy(y))

    def lorentzian_distance_func(self):
        lorentzian_mf_obj = LorentzManifold()
        return lambda x, y: lorentzian_mf_obj.lorentz_squared_distance(torch.from_numpy(x), torch.from_numpy(y))


class GeometricRepresentation(object):

    def __init__(self):
        logging.info(
            'Getting representative data points from different perspectives!')
        self.lr = 0.005
        self.gd_obj = self.get_gd_obj()
        self.color_options = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']

    def get_euclidean_centroids(self, distribution, point_num=100):
        # self.euclidean_sgd()
        return self.euclidean_k_means(distribution, point_num)

    def get_frechet_means(self, distribution, point_num=100):
        distance_func = lambda x, y: self.get_frechet_mean_distance(x, y)
        initial_centers = random.choices(distribution, k=point_num)
        frechet_means, clusters = self.get_points_by_common_kmeans(
                distance_func, gd_distribution=distribution, center_num=point_num, initial_centers=initial_centers)
        for f_mean in frechet_means:
            plt.scatter(f_mean[0], f_mean[1], c='r', marker='*')
        plt.title('Frechet means', fontsize='xx-large', fontweight='heavy')
        plt.show()
        return frechet_means

    def get_frechet_mean_distance(self, u, v):
        ed = euclidean_distance(u, v)
        return ed ** 2

    def get_poincare_centroids(self, distribution, point_num=100, need_normalization=False):
        poincare_mf_obj = PoincareManifold()
        distance_func = lambda x, y: poincare_mf_obj.distance(torch.from_numpy(x), torch.from_numpy(y))
        if need_normalization:
            distribution = self.get_normalized_distribution(distribution)
        initial_centers = random.choices(distribution, k=point_num)
        centroids, clusters = self.get_points_by_common_kmeans(
                distance_func, gd_distribution=distribution, center_num=point_num, initial_centers=initial_centers)
        for centroid in centroids:
            print(centroid)
            plt.scatter(centroid[0], centroid[1], c='r', marker='*')
        plt.title('Poincare centroids', fontsize='xx-large', fontweight='heavy')
        plt.show()
        return (centroids, clusters)

    # def get_poincare_centroids_v2(self, gd_distribution, clusters):
    #     poincare_mf_obj = PoincareManifold()
    #     centroids = []
    #     for cluster in clusters:
    #         sum_distance = 0
    #         for idx in cluster:
    #             sum_distance += poincare_mf_obj.distance(torch.from_numpy(current_point), torch.from_numpy(gd_distribution[idx]))

    def get_normalized_distribution(self, gd_distribution, show=True):
        scaler = sklearn.preprocessing.MinMaxScaler()
        normalized_gd_distribution = scaler.fit_transform(gd_distribution)
        if show:
            self._plot(normalized_gd_distribution)
        return normalized_gd_distribution

    def get_lorentzian_centroids(self, distribution, show=True, point_num=100, with_formula=False):
        lorentzian_mf_obj = LorentzManifold()
        distance_func = lambda x, y: lorentzian_mf_obj.lorentz_squared_distance(torch.from_numpy(x), torch.from_numpy(y))
        centroids, clusters = self.get_points_by_common_kmeans(distance_func, gd_distribution=distribution, center_num=point_num)
        if with_formula:
            centroids = self.get_points_by_formula(distribution, clusters, None)
        if show:
            for centroid in centroids:
                print(centroid)
                plt.scatter(centroid[0], centroid[1], c='r', marker='*', zorder=10)
            plt.title('Lorentzian centroids', fontsize='xx-large', fontweight='heavy')
            plt.show()
        return (centroids, clusters)

    def _plot(self, gd_distribution):
        last_index = 0
        size_list = self.gd_obj.get_size_list()
        for t in range(0, len(size_list)):
            start_index = last_index
            end_index = last_index + size_list[t]
            logging.info('start_index: {}, end_index: {}'.format(start_index, end_index))
            plt.scatter(
                gd_distribution[start_index: end_index, 0],
                gd_distribution[start_index: end_index, 1],
                c=self.color_options[t]
            )
            last_index += size_list[t]
        plt.title('Gaussian distribution', fontsize='xx-large', fontweight='heavy')

    def get_lorentzian_focal_points(self, distribution, point_num=100, with_formula=False):
        lorentzian_mf_obj = LorentzManifold()
        lorentzian_centroids, clusters = self.get_lorentzian_centroids(distribution, show=False, point_num=point_num)
        print('lorentzian_centroids: ', lorentzian_centroids)
        weights = self.get_weights(lorentzian_mf_obj, distribution, lorentzian_centroids, clusters)
        if with_formula:
            final_focal_points = self.get_points_by_formula(distribution, clusters, weights)
        else:
            final_focal_points = []
            for cluster in clusters:
                sub_distribution = np.array([distribution[idx] for idx in cluster])
                distance_func = lambda x, y: self.lorentzian_focal_distance_func(lorentzian_mf_obj, len(cluster), weights, x, y)
                initial_centers = [sub_distribution[0]]
                focal_points, clusters = self.get_points_by_common_kmeans(
                        distance_func, gd_distribution=sub_distribution, center_num=1, initial_centers=initial_centers)
                final_focal_points += focal_points

        print('final_focal_points: ', final_focal_points)

        for focal_point in final_focal_points:
            plt.scatter(focal_point[0], focal_point[1], c='r', marker='*')
        plt.title('Lorentzian focal points', fontsize='xx-large', fontweight='heavy')
        plt.show()
        return final_focal_points

    # TODO(Bad performance)
    def get_points_by_formula(self, gd_distribution, clusters, weights):
        final_points = []
        for cluster in clusters:
            sub_distribution = np.array([gd_distribution[idx] for idx in cluster])
            final_sub_distribution = []
            default_weight = 1 / sub_distribution.size
            for point in sub_distribution:
                if weights:
                    point_weight = weights.get(tuple(point))
                    if not point_weight:
                        print('the weight of point: {} lost, so set it 1/n'.format(tuple(point)))
                        point = default_weight * point # the default weight
                    else:
                        point = point_weight * point # lorentzian focal points weight
                else:
                    point = default_weight * point # lorentzian centroid weight(the same as the default weight now)
                final_sub_distribution.append(point)
            sum_vec = sum(np.array(final_sub_distribution))
            K = 150
            molecule = math.sqrt(K) * sum_vec
            denominator = -sum_vec[0] ** 2 + sum_vec[1] ** 2 # lorentzian centroids
            if weights: # lorentzian focal points
                denominator = abs(denominator)
            mu = molecule / denominator
            final_points.append(mu)
        return final_points

    def get_weights(self, lorentzian_mf_obj, gd_distribution, lorentzian_centroids, clusters):
        weights = {}
        point_centroid_map = {}
        sum_distance_map = {}
        for idx in range(0, len(lorentzian_centroids)):
            sum_distance = 0
            centroid = lorentzian_centroids[idx]
            for cluster_point_index in clusters[idx]:
                point = gd_distribution[cluster_point_index]
                l_dis = lorentzian_mf_obj.lorentz_squared_distance(torch.from_numpy(point), torch.tensor(centroid))
                point_centroid_map[tuple(point)] = {
                    'centroid': tuple(centroid),
                    'squared_distance': l_dis
                }
                sum_distance += l_dis
            sum_distance_map[tuple(centroid)] = sum_distance

        for point in gd_distribution:
            tuple_point= tuple(point)
            point_data = point_centroid_map.get(tuple_point)
            if not point_data:
                print('the weight of point: {} lost, so set it 1/n'.format(tuple_point))
                point_weight = 1 / gd_distribution.size
            else:
                point_weight = point_data['squared_distance'] / sum_distance_map[point_data['centroid']]
            weights[tuple_point] = point_weight
        return weights

    def lorentzian_focal_distance_func(self, lorentzian_mf_obj, distribution_size, weights, x, y):
        weight_x = weights.get(tuple(x))
        if not weight_x:
            print('the weight of point: [{}] lost!'.format(tuple(x)))
            weight_x = 1 / distribution_size
        return weight_x * lorentzian_mf_obj.lorentz_distance(torch.from_numpy(x), torch.from_numpy(y))

    # TODO
    def get_einstein_midpoints(self):
        raise NotImplementedError

    @abstractmethod
    def get_points_by_common_kmeans(self, distance_function, gd_distribution=None, center_num=3, initial_centers=[]):
        metric = distance_metric(type_metric.USER_DEFINED, func=distance_function)
        if not initial_centers:
            initial_centers = self.get_initial_centers(gd_distribution, center_num=center_num)

        # Create K-Means algorithm with specific distance metric
        kmeans_instance = kmeans(gd_distribution, initial_centers, metric=metric)
        # run cluster analysis and obtain results
        kmeans_instance.process()

        return (kmeans_instance.get_centers(), kmeans_instance.get_clusters())

    def euclidean_sgd(self):
        gd_distribution = self.gd_obj.get_guassian_distribution()
        size_list = self.gd_obj.get_size_list()
        total_groups_num = len(size_list)

        color_options = self.gd_obj.get_color_options()
        final_mu_map = self._get_final_mu_map(size_list, gd_distribution)
        last_index = 0
        for t in range(0, total_groups_num):
            start_index = last_index
            end_index = last_index + size_list[t]
            group_data = gd_distribution[start_index: end_index]
            last_index += size_list[t]
            plt.scatter(group_data[:, 0], group_data[:, 1], c=color_options[t])
        for mu_n, mu_data in final_mu_map.items():
            plt.scatter(mu_data[0], mu_data[1], c= 'r', marker='*')
        plt.title('Euclidean centroids by SGD', fontsize='xx-large', fontweight='heavy')
        plt.show()

    def _get_final_mu_map(self, size_list, gd_distribution):
        total_groups_num = len(size_list)
        initial_mu_map = {}
        last_index = 0
        for tg in range(0, total_groups_num):
            start_index = last_index
            end_index = last_index + size_list[tg]
            mu_name = 'mu_{}'.format(tg)
            group_data = gd_distribution[start_index: end_index]
            initial_mu_map[mu_name] = {
                'init_mu': random.choice(group_data),
                'group_data': group_data
            }
            last_index += size_list[tg]

        final_mu_map = {}
        # Iteration for each mu of each group
        for mu_na, data in initial_mu_map.items():
            mu = data['init_mu']
            for g_dis in data['group_data']:
                mu = mu + 2 * self.lr * (g_dis - mu)
            final_mu_map[mu_na] = mu

        return final_mu_map

    def euclidean_k_means(self, distribution, cluster_num=100):
        # Kmeans for each group
        cls = KMeans(n_clusters=cluster_num)
        cls_labels = cls.fit_predict(distribution)
        centers = cls.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='r', marker='*')
        plt.title('Euclidean centroids by k-means', fontsize='xx-large', fontweight='heavy')
        plt.show()
        clusters = [np.where(cls_labels == l)[0] for l in range(cluster_num)]
        return (centers, clusters)

    def coreset_by_k_greedy(self, distribution, point_num=100):
        kcg_obj = kCenterGreedy(distribution)
        idx_list = kcg_obj.select_batch_(model=None, already_selected=[], N=point_num)
        plt.scatter(distribution[idx_list][:, 0], distribution[idx_list][:, 1], c='r', marker='*')
        plt.title('Coreset by k-center greedy algorithm', fontsize='xx-large', fontweight='heavy')
        plt.show()
        return idx_list

    # TODO
    def robust_coreset_method(self):
        raise NotImplementedError

    def bayesian_coreset_method(self, distribution, distribution_type='guassian', gm_result=None, point_num=100, fill=False):
        bc_obj = BayesianCoreset()
        data_num = self._get_data_num(distribution_type)
        arguments = bc_obj.get_arguments(bc_obj, data_num, point_num)
        bc_coreset = bc_obj.construct_bayesian_coreset(distribution, arguments, distribution_type, gm_result)

        if fill:
            num_diff = point_num - len(bc_coreset)
            if num_diff:
                print('Get {} points with bayesian coreset selection'.format(len(bc_coreset)))
                diff_points = random.choices(distribution, k=num_diff)
                bc_coreset += diff_points
        return bc_coreset

    def _get_data_num(self, distribution_type):
        if distribution_type == 'guassian':
            return self.gd_obj.get_total_num()
        elif distribution_type == 'mnist':
            return 60000
        elif distribution_type == 'cifar10':
            return 50000
        elif distribution_type == 'spambase':
            return 4000

    def get_gd_obj(self):
        cfg = get_cfg_data()
        return GaussianDistribution(cfg['space-version']['type'], cfg['guassian-params']['items'])

    def get_initial_centers(self, gd_distribution, center_num):
        initial_centers = []
        size_list = self.gd_obj.get_size_list()
        total_size = sum(size_list)
        center_num_by_ratio = [int(math.ceil(size / total_size * center_num)) for size in size_list]
        gap = sum(center_num_by_ratio) - center_num
        final_center_num_by_ratio = center_num_by_ratio
        if gap > 0:
            final_center_num_by_ratio = center_num_by_ratio[0: -1]
            last_data = center_num_by_ratio[-1] - gap
            final_center_num_by_ratio.append(last_data)

        last_index = 0
        for t in range(0, len(size_list)):
            start_index = last_index
            end_index = last_index + size_list[t]
            print('start_index: {}, end_index: {}'.format(start_index, end_index))
            centers = random.choices(gd_distribution[start_index: end_index], k=final_center_num_by_ratio[t])
            initial_centers += centers
            last_index += size_list[t]
        return initial_centers


if __name__ == '__main__':
    gr_obj = GeometricRepresentation()
    cfg = get_cfg_data()
    gd_obj = GaussianDistribution(cfg['space-version']['type'], cfg['guassian-params']['items'])
    distribution = gd_obj.get_guassian_distribution()
    gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
    gr_obj.get_euclidean_centroids(distribution)
    gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
    gr_obj.get_frechet_means(distribution)
    gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
    gr_obj.coreset_by_k_greedy(distribution)
    gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
    gr_obj.bayesian_coreset_method(distribution)
    gr_obj.get_poincare_centroids(distribution)
    gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
    gr_obj.get_lorentzian_centroids(distribution)
    gd_obj.display_gd_distribution(distribution, show=False, scatt=True)
    gr_obj.get_lorentzian_focal_points(distribution)
    gc.collect()