#!/usr/bin/env python3
import logging
# logging.getLogger().setLevel(logging.INFO)
from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from utils import get_cfg_data


class GaussianDistribution(object):

    def __init__(self, space_type, guassian_params_list):
        self.space_type = space_type
        self.g_params_list = guassian_params_list
        self.color_options = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
        logging.info(
                'Creating guassian distribution in {} Paradigms'.format(self.space_type))

    def get_guassian_distribution(self, need_normalization=False):
        gp_len = len(self.g_params_list)
        if not self.is_params_ok():
            return
        if gp_len > len(self.color_options):
            logging.error(
                    'No available colors for distinguishing between different distributions, please provide more color options!')
            return
        lower_space_type = self.space_type.lower()
        if lower_space_type == 'euclidean':
            final_gd = self._build_euclidean_distribution()
            if need_normalization:
                final_gd = final_gd / 100
            return final_gd
        elif lower_space_type == 'non-euclidean':
            raise NotImplementedError

    def display_gd_distribution(self, final_gd, show=True, scatt=False):
        gp_len = len(self.g_params_list)
        size_list = self.get_size_list()
        last_index = 0
        for t in range(0, gp_len):
            start_index = last_index
            end_index = last_index + size_list[t]
            logging.info('start_index: {}, end_index: {}'.format(start_index, end_index))
            if scatt:
                plt.scatter(
                    final_gd[start_index: end_index, 0],
                    final_gd[start_index: end_index, 1],
                    c=self.color_options[t]
                )
            last_index += size_list[t]
        if show:
            plt.title('Gaussian distribution', fontsize='xx-large', fontweight='heavy')
            plt.show()

    # Map data points from euclidean space to lorentz model
    def _exp_to_lorentz(self):
        pass

    # Map data points from riemannian manifold to euclidean sapce
    def _log_to_euclidean(self):
        pass

    def _build_euclidean_distribution(self):
        np.random.seed(1)
        first_param = self._get_final_param(self.g_params_list[0])
        final_gd = np.random.normal(first_param['loc'], first_param['scale'], first_param['size'])
        # display datapoints of the first group
        # self._show_prob_density(first_param, final_gd)
        for g_param in self.g_params_list[1:]:
            final_param = self._get_final_param(g_param)
            gd = np.random.normal(final_param['loc'], final_param['scale'], final_param['size'])
            # display datapoints from the second group
            # self._show_prob_density(final_param, gd)
            final_gd = np.concatenate((final_gd, gd))
        return final_gd

    def _convert_to_np_array(self, param_in):
        if isinstance(param_in, int) or isinstance(param_in, float):
            return param_in
        return np.fromstring(param_in, dtype=int, sep=' ')

    def _get_final_param(self, g_param):
        return {
            'loc': self._convert_to_np_array(g_param['loc']),
            'scale': self._convert_to_np_array(g_param['scale']),
            'size': self._convert_to_np_array(g_param['size']),
        }

    # TODO(Weixin Bu): We are supposed to modify the formula.
    def _show_prob_density(self, final_param, gd):
        _loc = final_param['loc']
        _scale = final_param['scale']
        if isinstance(_loc, np.ndarray):
            low_bound_loc = _loc[0]
            high_bound_loc = _loc[1]
            _loc = (low_bound_loc + high_bound_loc) / 2
        if isinstance(_scale, np.ndarray):
            low_bound_scale = _scale[0]
            high_bound_scale = _scale[1]
            _scale = (low_bound_scale + high_bound_scale) / 2
        final_loc = _loc
        final_scale = _scale
        count, bins, ignored = plt.hist(gd, 30, density=True)
        formula = 1 / (final_scale * np.sqrt(2 * np.pi)) * np.exp( - (bins - final_loc)**2 / (2 * final_scale **2))
        plt.plot(bins, formula, linewidth=2, color='r')
        plt.show()

    @abstractmethod
    def get_size_list(self):
        size_list = []
        for ds in list(
                map(lambda param: self._convert_to_np_array(param['size']), self.g_params_list)):
            if isinstance(ds, np.ndarray):
                size_list.append(ds[0])
            elif isinstance(ds, int):
                size_list.append(ds)
        return size_list

    def get_total_num(self):
        return sum(self.get_size_list())

    def is_params_ok(self):
        if not self.space_type:
            logging.error('Fail to get space_type!')
            return False
        if not self.g_params_list:
            logging.error('Fail to get g_params_list!')
            return False
        return True

    def get_color_options(self):
        return self.color_options

    # high dimensional
    def get_hd_gaussian_distribution(self):
        mu = np.ones(100) * 5
        cov = np.eye(100) * np.random.rand(100)
        d1 = np.random.multivariate_normal(mu, cov, 100)

        mu = np.ones(100) * 3
        cov = np.eye(100) * np.random.rand(100)
        d2 = np.random.multivariate_normal(mu, cov, 100)
        return np.concatenate((d1, d2), axis=0)


if __name__ == '__main__':
    cfg = get_cfg_data()
    gd_obj = GaussianDistribution(cfg['space-version']['type'], cfg['guassian-params']['items'])
    gd_distribution = gd_obj.get_guassian_distribution()
    gd_obj.display_gd_distribution(gd_distribution, show=True, scatt=True)
