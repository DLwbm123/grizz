#!/usr/bin/env python3
from abc import ABC, abstractmethod
import numpy as np
import logging
import numpy.matlib as nm
import torch
from scipy.spatial.distance import pdist, squareform
from torch.optim.optimizer import Optimizer, required

from evaluation import MaximumMeanDiscrepancy, ResultDisplay
from manifold import PoincareManifold
from real_dataset import RealisticDatasets, DimensionalityReduction, Mixture
from representation import GeometricRepresentation


class OptimizerBase(ABC):

    def __init__(self):
        logging.info('The base optimizer')

    def __call__(self, params, params_grad, params_name):
        return self.update(params, params_grad, params_name)

    @abstractmethod
    def update(self, params, params_grad, params_name):
        raise NotImplementedError


class SGD(OptimizerBase):

    def __init__(self, lr=0.005):
        logging.info('Stochastic Gradient Descent')
        super(SGD, self).__init__()
        self.lr = lr
        self.cache = {}

    def update(self, params, params_grad, params_name):
        updated_value = self.lr * params_grad
        return params - updated_value

    @property
    def hyperparams(self):
        return {
            'optimizer': 'SGD',
            'lr': self.lr
        }


class RiemannianSGD(object):
    pass


class RiemannianSGDV2(Optimizer):

    def __init__(
            self, size, dim, sparse, params,
            lr=required, rgrad=required, expm=required
    ):
        logging.info('Riemannian Stochastic Gradient Descent')
        self.manifold = PoincareManifold
        self.size = size
        self.dim = dim
        self.sparse = sparse
        defaults = {
            'lr': lr,
            'rgrad': rgrad,
            'expm': expm,
        }
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None, counts=None, **kwargs):
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                lr = lr or group['lr']
                rgrad = group['rgrad']
                expm = group['expm']

                if p.grad is None:
                    continue
                d_p = p.grad.data
                # make sure we have no duplicates in sparse tensor
                if d_p.is_sparse:
                    d_p = d_p.coalesce()
                d_p = rgrad(p.data, d_p)
                d_p.mul_(-lr)
                expm(p.data, d_p)

        return loss

    def get_optimal_params(self):
        al_lt = self.manifold.allocate_lt(self.size, self.dim, self.sparse)
        return [{
            'params': al_lt.parameters(),
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
            'ptransp': self.manifold.ptransp,
        }]


class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A

    def dlnprob(self, theta):
        return -1 * np.matmul(theta - nm.repmat(self.mu, theta.shape[0], 1), self.A)


class BayesianLR(object):
    def __init__(self, X, Y, batchsize=100, a0=1, b0=0.01):
        self.X, self.Y = X, Y
        # TODO. Y in \in{+1, -1}
        self.batchsize = min(batchsize, X.shape[0])
        self.a0, self.b0 = a0, b0

        self.N = X.shape[0]
        self.permutation = np.random.permutation(self.N)
        self.iter = 0

    def dlnprob(self, theta):

        if self.batchsize > 0:
            batch = [i % self.N for i in range(self.iter * self.batchsize, (self.iter + 1) * self.batchsize)]
            ridx = self.permutation[batch]
            self.iter += 1
        else:
            ridx = np.random.permutation(self.X.shape[0])

        Xs = self.X[ridx, :]
        Ys = self.Y[ridx]

        w = theta[:, :-1]  # logistic weights
        alpha = np.exp(theta[:, -1])  # the last column is logalpha
        d = w.shape[1]

        wt = np.multiply((alpha / 2), np.sum(w ** 2, axis=1))

        coff = np.matmul(Xs, w.T)
        y_hat = 1.0 / (1.0 + np.exp(-1 * coff))

        dw_data = np.matmul(((nm.repmat(np.vstack(Ys), 1, theta.shape[0]) + 1) / 2.0 - y_hat).T, Xs)  # Y \in {-1,1}
        dw_prior = -np.multiply(nm.repmat(np.vstack(alpha), 1, d), w)
        dw = dw_data * 1.0 * self.X.shape[0] / Xs.shape[0] + dw_prior  # re-scale

        dalpha = d / 2.0 - wt + (self.a0 - 1) - self.b0 * alpha + 1  # the last term is the jacobian term

        return np.hstack([dw, np.vstack(dalpha)])  # % first order derivative

    def evaluation(self, theta, X_test, y_test):
        theta = theta[:, :-1]
        M, n_test = theta.shape[0], len(y_test)

        prob = np.zeros([n_test, M])
        for t in range(M):
            coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
            prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))

        prob = np.mean(prob, axis=1)
        acc = np.mean(prob > 0.5)
        llh = np.mean(np.log(prob))
        return [acc, llh]


class SVGD(object):

    def __init__(self):
        logging.info('Stein Variational Gradient Descent')

    def svgd_kernel(self, theta, h=-1):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

        # compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h ** 2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h ** 2)
        return (Kxy, dxkxy)

    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):
            if debug and (iter + 1) % 1000 == 0:
                print('iter ' + str(iter + 1))

            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h=-1)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

            # adagrad
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

        return theta


class GdResultDisplay(object):

    def __init__(self):
        logging.info('Realize gradient descent result display')

    def display(self, distribution, distribution_type, n, model, mmd_map):
        lr = 0.005
        euclidean_centroids, clusters = gr_obj.get_euclidean_centroids(distribution, point_num=n)
        et = 0
        sgd_points = []
        for cluster in clusters:
            e_mu = euclidean_centroids[et]
            points = distribution[cluster]
            for point in points:
                e_mu = e_mu + 2 * lr * (point - e_mu)
            sgd_points.append(e_mu)
            et += 1
        euclidean_centroids_mmd = mmd_obj.forward(torch.tensor(distribution), torch.tensor(sgd_points))
        print('--------------------------------------------------------------------------------')
        print('Maximum Mean Discrepancy: {} with n: {} for euclidean centroids(SGD)'.format(euclidean_centroids_mmd.item(), n))
        mmd_map_k = 'SGD_{}'.format(distribution_type)
        mmd_map.setdefault(mmd_map_k, [])
        mmd_map[mmd_map_k].append((n, euclidean_centroids_mmd.item()))

        poincare_centroids, poincare_clusters = gr_obj.get_poincare_centroids(distribution, point_num=n)
        rt = 0
        rsgd_points = []
        for poincare_cluster in poincare_clusters:
            r_mu = poincare_centroids[rt]
            points = distribution[poincare_cluster]
            for point in points:
                star = 1 + 2 * (np.linalg.norm(point - r_mu) ** 2) / (
                            (1 - (np.linalg.norm(point) ** 2)) * (1 - (np.linalg.norm(r_mu) ** 2)))
                molecule = (-2 * (point - r_mu) * (1 - (np.linalg.norm(r_mu) ** 2)) + (
                            np.linalg.norm(point - r_mu) ** 2) * 2 * r_mu)
                denominator = (1 - (np.linalg.norm(r_mu) ** 2)) ** 2
                r_mu = r_mu - lr * (1 / np.sqrt(star ** 2 - 1)) * (2 / (1 - (np.linalg.norm(point) ** 2))) * molecule / denominator
            rsgd_points.append(r_mu)
            rt += 1
        poincare_centroids_mmd = mmd_obj.forward(torch.tensor(distribution), torch.tensor(rsgd_points))
        print('--------------------------------------------------------------------------------')
        print('Maximum Mean Discrepancy: {} with n: {} for euclidean centroids(RGD)'.format(poincare_centroids_mmd.item(), n))
        mmd_map_k = 'RGD_{}'.format(distribution_type)
        mmd_map.setdefault(mmd_map_k, [])
        mmd_map[mmd_map_k].append((n, poincare_centroids_mmd.item()))

        updated_centroids = svgd_obj.update(euclidean_centroids, model.dlnprob, n_iter=10000, stepsize=0.01)
        updated_centroids_mmd = mmd_obj.forward(torch.tensor(distribution), torch.tensor(updated_centroids))
        print('--------------------------------------------------------------------------------')
        print('Maximum Mean Discrepancy: {} with n: {} for euclidean centroids(SVGD)'.format(updated_centroids_mmd.item(), n))
        mmd_map_k = 'SVGD_{}'.format(distribution_type)
        mmd_map.setdefault(mmd_map_k, [])
        mmd_map[mmd_map_k].append((n, updated_centroids_mmd.item()))


if __name__ == '__main__':
    rd = ResultDisplay()
    rd_obj = RealisticDatasets()
    result_display_obj = ResultDisplay()
    dr_obj = DimensionalityReduction()
    gr_obj = GeometricRepresentation()
    mmd_obj = MaximumMeanDiscrepancy()
    svgd_obj = SVGD()
    gdrd_obj = GdResultDisplay()

    X, y = rd_obj.get_mnist_data()
    normalized_X = gr_obj.get_normalized_distribution(X, show=False)
    newX = dr_obj.pca_dr(normalized_X)
    newY = [int(val) for _, val in y.iteritems()]

    mixture_obj = Mixture()
    _mean, _cov = mixture_obj.get_gaussian_mixture_result(newX[0: 60000])
    mnist_model = MVN(_mean[0], _cov[0])

    print('Start gradient descents for MNIST distribution')
    mnist_mmd_map = {}
    for n in range(50, 600, 50):
        gdrd_obj.display(newX[0: 60000], 'mnist', n, mnist_model, mnist_mmd_map)

    title = 'Maximum Mean Discrepancy with different batch settings'
    y_label = 'Maximum Mean Discrepancy'
    rd.plot_result(title, mnist_mmd_map, y_label)
    #
    # print('Start gradient descents for Cifar10 distribution')
    # cifar10_trainloader = rd_obj.get_cifar10_trainloader()
    # # dimensional reduction realized in inner part of `get_cifar10_data`
    # cifar10_output, cifar10_labels = rd_obj.get_cifar10_data(cifar10_trainloader)
    # _cifar10_mean, _cifar10_cov = mixture_obj.get_gaussian_mixture_result(cifar10_output[0: 45000])
    # cifar10_model = MVN(_cifar10_mean[0], _cifar10_cov[0])
    # cifar10_mmd_map = {}
    # for n in range(50, 650, 50):
    #     gdrd_obj.display(cifar10_output[0: 45000], 'cifar10', n, cifar10_model, cifar10_mmd_map)
    #
    # title = 'Maximum Mean Discrepancy with different batch settings'
    # y_label = 'Maximum Mean Discrepancy'
    # rd.plot_result(title, cifar10_mmd_map, y_label)