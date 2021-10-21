#!/usr/bin/env python3
import logging
import random

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from bayesiancoresets import HilbertCoreset
from bayesiancoresets import BlackBoxProjector
from gaussian import GaussianDistribution
from utils import get_cfg_data
import examples.common.model_lr as m_lr
import scipy.linalg as sl
import bayesiancoresets.util as bc_util
import os, sys
import argparse
import time
import bayesiancoresets as bc
# make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import examples.common.model_gaussian as gaussian


class SimpleLRBayesianCoreset(object):

    def __init__(self, gd_distribution):
        self.coreset_threshold = 200
        self.gd_distribution = gd_distribution
        logging.info('Construct a simple bayesian coreset!')

    def construct_bayesian_coreset(self):
        Z = self.gd_distribution
        # Here we use the laplace approximation of the posterior
        # first, optimize the log joint to find the mode:
        res = minimize(lambda mu: -m_lr.log_joint(Z, mu, np.ones(Z.shape[0]))[0], Z.mean(axis=0),
                       jac=lambda mu: -m_lr.grad_th_log_joint(Z, mu, np.ones(Z.shape[0]))[0, :])
        # then find a quadratic expansion around the mode, and assume the distribution is Gaussian
        mu = res.x
        cov = -np.linalg.inv(m_lr.hess_th_log_joint(Z, mu, np.ones(Z.shape[0]))[0, :, :])
        projection_dim = 500  # random projection dimension
        sampler = lambda sz, w, p: np.atleast_2d(np.random.multivariate_normal(mu, cov, sz))
        projector = BlackBoxProjector(sampler, projection_dim, m_lr.log_likelihood)
        coreset = HilbertCoreset(Z, projector)
        coreset.build(self.coreset_threshold)  # build the coreset to size M with at most M iterations
        wts, pts, idcs = coreset.get()
        plt.scatter(self.gd_distribution[idcs][:, 0], self.gd_distribution[idcs][:, 1], c='r', marker='*')
        plt.title('Bayesian coreset', fontsize='xx-large', fontweight='heavy')
        plt.show()
        return idcs


class BayesianCoreset(object):

    def __init__(self):
        cfg = get_cfg_data()
        self.gd_obj = GaussianDistribution(cfg['space-version']['type'], cfg['guassian-params']['items'])
        self.g_params_list = cfg['guassian-params']['items']
        logging.info('Construct a Gaussian bayesian coreset!')

    def construct_bayesian_coreset(self, distribution, arguments, distribution_type, gm_result):
        #######################################
        #######################################
        ## Step 0: Setup
        #######################################
        #######################################
        np.random.seed(arguments.trial)
        bc_util.set_verbosity(arguments.verbosity)
        Ms = arguments.coreset_size_max
        # w = []
        # p = []

        final_coreset_points = []
        if distribution_type == 'guassian':
            last_index = 0
            for g_param in self.g_params_list:
                loc_data = np.array([float(val) for val in g_param['loc'].split(' ')])
                scale_data = np.array([float(val) for val in g_param['scale'].split(' ')])
                size_data = [int(val) for val in g_param['size'].split(' ')]
                #######################################
                #######################################
                ## Step 1: Generate a Synthetic Dataset
                #######################################
                #######################################
                mu0 = np.array(loc_data)
                Sig0 = np.array([
                    [scale_data[0], 0],
                    [0, scale_data[1]]
                ])
                Sig = np.array([
                    [scale_data[0], 0],
                    [0, scale_data[1]]
                ])

                # these are computed
                Sig0inv = np.linalg.inv(Sig0)
                Siginv = np.linalg.inv(Sig)
                LSigInv = np.linalg.cholesky(Siginv)  # Siginv = LL^T, L Lower tri
                USig = sl.solve_triangular(
                        LSigInv, np.eye(LSigInv.shape[0]), lower=True, overwrite_b=True, check_finite=False).T  # Sig = UU^T, U upper tri
                # th = np.ones(arguments.data_dim)
                th = np.array(loc_data)
                logdetSig = np.linalg.slogdet(Sig)[1]
                #######################################
                #######################################
                ## Step 2: Calculate Likelihoods/Projectors
                #######################################
                #######################################
                # print('Computing true posterior')
                start_index = last_index
                end_index = last_index + size_data[0]
                x = distribution[start_index: end_index]
                last_index += size_data[0]
                # x = np.random.multivariate_normal(th, Sig, arguments.data_num)
                mup, USigp, LSigpInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
                Sigp = USigp.dot(USigp.T)
                SigpInv = LSigpInv.dot(LSigpInv.T)

                sub_coreset = self._get_bayesian_sub_coreset(
                        Siginv, logdetSig, mup, USigp, x, mu0, Sig0inv, LSigInv, Ms, arguments)
                final_coreset_points += sub_coreset
        else:
            _mean = gm_result[0][0]
            _cov = gm_result[1][0]
            mu0 = _mean
            Sig0 = _cov
            Sig = _cov
            # these are computed
            Sig0inv = np.linalg.inv(Sig0)
            Siginv = np.linalg.inv(Sig)
            LSigInv = np.linalg.cholesky(Siginv)  # Siginv = LL^T, L Lower tri
            USig = sl.solve_triangular(
                LSigInv, np.eye(LSigInv.shape[0]), lower=True, overwrite_b=True,
                check_finite=False).T  # Sig = UU^T, U upper tri
            # th = np.ones(arguments.data_dim)
            th = gm_result[0]
            logdetSig = np.linalg.slogdet(Sig)[1]
            #######################################
            #######################################
            ## Step 2: Calculate Likelihoods/Projectors
            #######################################
            #######################################
            # print('Computing true posterior')
            x = distribution
            mup, USigp, LSigpInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
            Sigp = USigp.dot(USigp.T)
            SigpInv = LSigpInv.dot(LSigpInv.T)

            sub_coreset = self._get_bayesian_sub_coreset(
                Siginv, logdetSig, mup, USigp, x, mu0, Sig0inv, LSigInv, Ms, arguments)
            final_coreset_points += sub_coreset

        # num_diff = Ms - len(final_coreset_points)
        # random.choices(distribution, k=num_diff)
        for point in final_coreset_points:
            plt.scatter(
                point[0], point[1], c='r', marker='*'
            )
        plt.title('Bayesian coreset', fontsize='xx-large', fontweight='heavy')
        plt.show()
        return final_coreset_points

    def _get_bayesian_sub_coreset(self, Siginv, logdetSig, mup, USigp, x, mu0, Sig0inv, LSigInv, Ms, arguments):
        # create the log_likelihood function
        # print('Creating log-likelihood function')
        log_likelihood = lambda x, th: gaussian.log_likelihood(x, th, Siginv, logdetSig)

        # print('Creating gradient log-likelihood function')
        grad_log_likelihood = lambda x, th: gaussian.grad_x_log_likelihood(x, th, Siginv)

        # print('Creating tuned projector for Hilbert coreset construction')
        # create the sampler for the "optimally-tuned" Hilbert coreset
        sampler_optimal = lambda n, w, pts: mup + np.random.randn(n, mup.shape[0]).dot(USigp.T)
        prj_optimal = bc.BlackBoxProjector(sampler_optimal, arguments.proj_dim, log_likelihood, grad_log_likelihood)

        # print('Creating untuned projector for Hilbert coreset construction')
        # create the sampler for the "realistically-tuned" Hilbert coreset
        xhat = x[np.random.randint(0, x.shape[0], int(np.sqrt(x.shape[0]))), :]
        muhat, USigHat, LSigHatInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, xhat, np.ones(xhat.shape[0]))
        sampler_realistic = lambda n, w, pts: muhat + np.random.randn(n, muhat.shape[0]).dot(USigHat.T)
        prj_realistic = bc.BlackBoxProjector(sampler_realistic, arguments.proj_dim, log_likelihood,
                                             grad_log_likelihood)

        # print('Creating black box projector')

        def sampler_w(n, wts, pts):
            if wts is None or pts is None or pts.shape[0] == 0:
                wts = np.zeros(1)
                pts = np.zeros((1, mu0.shape[0]))
            muw, USigw, _ = gaussian.weighted_post(mu0, Sig0inv, Siginv, pts, wts)
            return muw + np.random.randn(n, muw.shape[0]).dot(USigw.T)

        prj_bb = bc.BlackBoxProjector(sampler_w, arguments.proj_dim, log_likelihood, grad_log_likelihood)

        # print('Creating exact projectors')

        # TODO need to fix all the transposes in this...
        class GaussianProjector(bc.Projector):
            def project(self, pts, grad=False):
                nu = (pts - self.muw).dot(LSigInv)
                PsiL = LSigInv.T.dot(self.USigw)
                Psi = PsiL.dot(PsiL.T)
                nu = np.hstack(
                    (nu.dot(PsiL),
                     np.sqrt(0.5 * np.trace(np.dot(Psi.T, Psi))) * np.ones(nu.shape[0])[:, np.newaxis]))
                nu *= np.sqrt(nu.shape[1])
                if not grad:
                    return nu
                else:
                    gnu = np.hstack((LSigInv.dot(PsiL), np.zeros(pts.shape[1])[:, np.newaxis])).T
                    gnu = np.tile(gnu, (pts.shape[0], 1, 1))
                    gnu *= np.sqrt(gnu.shape[1])
                    return nu, gnu

            def update(self, wts=None, pts=None):
                if wts is None or pts is None or pts.shape[0] == 0:
                    wts = np.zeros(1)
                    pts = np.zeros((1, mu0.shape[0]))
                self.muw, self.USigw, self.LSigwInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, pts, wts)

        prj_optimal_exact = GaussianProjector()
        prj_optimal_exact.update(np.ones(x.shape[0]), x)
        prj_realistic_exact = GaussianProjector()
        prj_realistic_exact.update(np.ones(xhat.shape[0]), xhat)

        #######################################
        #######################################
        ## Step 3: Construct Coreset
        #######################################
        #######################################

        # print('Creating coreset construction objects')
        # create coreset construction objects
        sparsevi_exact = bc.SparseVICoreset(x, GaussianProjector(), opt_itrs=arguments.opt_itrs,
                                            step_sched=eval(arguments.step_sched))
        sparsevi = bc.SparseVICoreset(x, prj_bb, opt_itrs=arguments.opt_itrs, step_sched=eval(arguments.step_sched))
        giga_optimal = bc.HilbertCoreset(x, prj_optimal)
        giga_optimal_exact = bc.HilbertCoreset(x, prj_optimal_exact)
        giga_realistic = bc.HilbertCoreset(x, prj_realistic)
        giga_realistic_exact = bc.HilbertCoreset(x, prj_realistic_exact)
        unif = bc.UniformSamplingCoreset(x)

        algs = {'SVI-EXACT': sparsevi_exact,
                'SVI': sparsevi,
                'GIGA-OPT': giga_optimal,
                'GIGA-OPT-EXACT': giga_optimal_exact,
                'GIGA-REAL': giga_realistic,
                'GIGA-REAL-EXACT': giga_realistic_exact,
                'US': unif}

        alg = algs[arguments.alg]
        # print('Building coreset')
        t_build = 0
        # print('M = ' + str(Ms) + ': coreset construction, ' + arguments.alg + ' ' + str(arguments.trial))
        t0 = time.process_time()
        itrs = Ms
        alg.build(itrs)
        t_build += time.process_time() - t0
        wts, pts, idcs = alg.get()
        # store weights/pts/runtime
        # w.append(wts)
        # p.append(pts)
        sub_coreset = [x[idx] for idx in idcs]
        return sub_coreset

    def get_arguments(self, bc_obj, data_num, point_num=100):
        parser = argparse.ArgumentParser(
                description="Runs Riemannian linear regression (employing coreset contruction) on the specified dataset")
        subparsers = parser.add_subparsers(help='sub-command help')
        run_subparser = subparsers.add_parser('run', help='Runs the main computational code')
        run_subparser.set_defaults(func=bc_obj.construct_bayesian_coreset)

        parser.add_argument('--data_num', type=int, default=data_num, help='Dataset size/number of examples')
        parser.add_argument('--data_dim', type=int, default=2,
                            help="The dimension of the multivariate normal distribution to use for this experiment")
        parser.add_argument('--alg', type=str, default='GIGA-OPT',
                            choices=['SVI', 'SVI-EXACT', 'GIGA-OPT', 'GIGA-OPT-EXACT', 'GIGA-REAL', 'GIGA-REAL-EXACT',
                                     'US'],
                            help="The name of the coreset construction algorithm to use")
        parser.add_argument("--proj_dim", type=int, default=2000,
                            help="The number of samples taken when discretizing log likelihoods for these experiments")
        parser.add_argument('--coreset_size_max', type=int, default=point_num, help="The maximum coreset size to evaluate")
        parser.add_argument('--opt_itrs', type=int, default=200,
                            help="Number of optimization iterations (for methods that use iterative weight refinement)")
        parser.add_argument('--step_sched', type=str, default="lambda i : 1./(1+i)",
                            help="Optimization step schedule (for methods that use iterative weight refinement); entered as a python lambda expression surrounded by quotes")
        parser.add_argument('--trial', type=int, default=1,
                            help="The trial number - used to initialize random number generation (for replicability)")
        parser.add_argument('--verbosity', type=str, default="error",
                            choices=['error', 'warning', 'critical', 'info', 'debug'],
                            help="The verbosity level.")

        arguments = parser.parse_args()
        return arguments


if __name__ == '__main__':
    bc_obj = BayesianCoreset()
    cfg = get_cfg_data()
    gd_obj = GaussianDistribution(cfg['space-version']['type'], cfg['guassian-params']['items'])
    distribution = gd_obj.get_guassian_distribution()
    gd_obj.display_gd_distribution(distribution, show=False, scatt=True)

    parser = argparse.ArgumentParser(
        description="Runs Riemannian linear regression (employing coreset contruction) on the specified dataset")
    subparsers = parser.add_subparsers(help='sub-command help')
    run_subparser = subparsers.add_parser('run', help='Runs the main computational code')
    run_subparser.set_defaults(func=bc_obj.construct_bayesian_coreset)

    parser.add_argument('--data_num', type=int, default='6000', help='Dataset size/number of examples')
    parser.add_argument('--data_dim', type=int, default='2',
                        help="The dimension of the multivariate normal distribution to use for this experiment")
    parser.add_argument('--alg', type=str, default='GIGA-OPT',
                        choices=['SVI', 'SVI-EXACT', 'GIGA-OPT', 'GIGA-OPT-EXACT', 'GIGA-REAL', 'GIGA-REAL-EXACT',
                                 'US'],
                        help="The name of the coreset construction algorithm to use")
    parser.add_argument("--proj_dim", type=int, default=2000,
                        help="The number of samples taken when discretizing log likelihoods for these experiments")
    parser.add_argument('--coreset_size_max', type=int, default=500, help="The maximum coreset size to evaluate")
    parser.add_argument('--opt_itrs', type=int, default=200,
                        help="Number of optimization iterations (for methods that use iterative weight refinement)")
    parser.add_argument('--step_sched', type=str, default="lambda i : 1./(1+i)",
                        help="Optimization step schedule (for methods that use iterative weight refinement); entered as a python lambda expression surrounded by quotes")
    parser.add_argument('--trial', type=int, default=1,
                        help="The trial number - used to initialize random number generation (for replicability)")
    parser.add_argument('--verbosity', type=str, default="error",
                        choices=['error', 'warning', 'critical', 'info', 'debug'],
                        help="The verbosity level.")

    arguments = parser.parse_args()

    print('coreset_size_max: ', arguments.coreset_size_max)

    results = bc_obj.construct_bayesian_coreset(distribution, arguments, 'guassian', None)
    print(results)
