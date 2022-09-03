import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
import scipy.stats as stats
from scipy.linalg import cho_solve
from collections import namedtuple
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from util import *


class MultiGPR:
    # Conditionally independent multi-output Gaussian Process Regression from Deisenroth et al. (2017)
    # This is a base model for components in PILCO
    def __init__(self, input_dim, output_dim, kernel=None, **kwargs):
        self.input_dim = input_dim      # (D)
        self.output_dim = output_dim    # (F)
        if kernel is None:
            kernel = C(1.0, (1e-2, 1e2)) * RBF(np.ones(self.input_dim)) + W(1e-1, (1e-5, 1.))
        else:
            pass

        self.gps = {}
        for i in range(self.output_dim):
            self.gps[i] = GaussianProcessRegressor(kernel=kernel.clone_with_theta(kernel.theta), **kwargs)

        # find the kernel key names in case of arbitrary kernel
        for k in kernel.hyperparameters:
            if 'length_scale' in k.name:
                self.length_scale_key = k.name
            elif 'constant_value' in k.name:
                self.constant_value_key = k.name
            else:
                pass
        
    def fit(self, X, y):
        assert X.shape[1] == self.input_dim
        assert y.shape[1] == self.output_dim

        self.X_train_ = X.copy()    # (N, D)    
        self.y_train_ = y.copy()    # (N, F)

        for i in range(self.output_dim):
            self.gps[i].fit(X, y[:, i])
                
    def predict(self, x, deterministic=False):
        # Input: scipy.stats.mltivariate_normal class object (x)
        # Output: scipy.stats.mltivariate_normal class object (m_new, S_new), input-output covariance matrix (C)
        m = x.mean  # (D)
        S = x.cov   # (D, D)

        nu = self.X_train_ - m[None, :]    # (N, D)
        L = np.eye(self.input_dim) * self.squared_length[:, np.newaxis, :]  # (F, D, D)
        iL = np.eye(self.input_dim) / self.squared_length[:, np.newaxis, :] # (F, D, D)

        iT = np.linalg.inv(S + L)   # (F, D, D)
        q = np.exp(np.log(self.const)[:, None] - np.product(np.linalg.slogdet(S @ iL + np.eye(self.input_dim)), axis=0)[:, None] / 2 - np.diagonal(nu @ iT @ nu.T, 0, 1, 2) / 2)
        m_new = np.einsum('ij,ji->i', self.alpha_, q.T)
        C = np.einsum('fdn, fn->fd', S @ iT @ nu.T, q * self.alpha_).T    # (D, D) x (F, D, D) x (D, N) => (F, D, N) x (F, N) => (F, D)

        R = S @ (iL[:, None] + iL[None, :]) + np.eye(self.input_dim)    # (F, F, D, D)
        Z = nu @ iL     # (F, N, D)
        nuiL = np.diagonal(squared_maha(nu, Q=iL), 0, 1, 2)
        ln_Q = np.tensordot(np.log(self.const)[:, None] + np.log(self.const)[None, :] - np.product(np.linalg.slogdet(R), 0) / 2, np.ones((self.X_train_.shape[0], self.X_train_.shape[0])), 0) \
            - nuiL[None, :, :, None] / 2 - nuiL[:, None, None, :] / 2 + squared_maha(Z, -Z, np.linalg.solve(R, S)) / 2
        Q = np.exp(ln_Q)    # (F, F, N, N)
        S_new = np.einsum('ip, ijpq, jq->ij', self.alpha_, Q, self.alpha_) - np.outer(m_new, m_new)
        if deterministic:
            S_new += np.eye(self.output_dim) * 1e-6
        else:
            S_new += np.diag(self.const - [cho_solve((self.L_[i], 1), Q[i, i]).trace() for i in range(self.output_dim)])

        return stats.multivariate_normal(m_new, S_new, allow_singular=True), C
    
    @property
    def squared_length(self):
        return np.stack(2 * np.square(self.gps[i].kernel_.get_params()[self.length_scale_key]) for i in range(self.output_dim)) # (F, D)

    @property
    def const(self):
        return np.stack(self.gps[i].kernel_.get_params()[self.constant_value_key] for i in range(self.output_dim))  #(F)
    
    @property
    def alpha_(self):
        return np.stack(self.gps[i].alpha_ for i in range(self.output_dim)) #(F, N)

    @property
    def L_(self):
        return np.stack(self.gps[i].L_ for i in range(self.output_dim)) #(F, N, N)


class RBFPolicy(MultiGPR):
    def __init__(self, state_dim, control_dim, level=20, u_max=1.0, kernel=None, **kwargs):
        if kernel is None:
            kernel = C(1.0, 'fixed') * RBF(np.exp(np.random.randn(state_dim)), 'fixed') # use a fixed kernel
        else:
            kernel = kernel
        super().__init__(state_dim, control_dim, kernel=kernel, alpha=1e-4, **kwargs)
        self.level = level
        self.u_max = u_max

        # initialize the policy with random parameters
        self.theta_dims = np.array([[self.level, self.input_dim], [self.level, self.output_dim], [self.output_dim, self.input_dim]])  # (M, T, legnth_scale)
        self.theta_len = np.multiply(*self.theta_dims.T).sum()
        self.randomize()
        
    def randomize(self):
        print('Randomizing the policy')
        M = np.random.randn(*self.theta_dims[0])
        T = self.u_max / 10 * np.random.randn(*self.theta_dims[1])
        L = np.random.randn(*self.theta_dims[2])
        theta = np.hstack((M.ravel(), T.ravel(), L.ravel()))
        self.update(theta)
    
    def update(self, theta):
        self.theta = theta
        # theta is a vector, in which length_scale values are taken logarithm
        M, T, L = np.split(self.theta, np.cumsum(np.multiply(*self.theta_dims.T))[:-1])
        M = M.reshape(*self.theta_dims[0])
        T = T.reshape(*self.theta_dims[1])
        L = np.exp(L.reshape(*self.theta_dims[2]))
        for i in range(self.output_dim):
            self.gps[i].kernel.set_params(k2__length_scale=L[i])
        self.fit(M, T)
    
    def squash_sin(self, x):
        # a sine function squashing restricts the preliminary policy output to [-1, 1]
        m = x.mean
        S = x.cov

        m_new = self.u_max * np.exp(-np.diag(S) / 2) * np.sin(m)    # 
        lq = -(np.diag(S)[:, None] + np.diag(S)[None, :]) / 2
        q = np.exp(lq)
        S_new = (np.exp(lq + S) - q) * np.cos(m[:, None] - m[None, :]) - (np.exp(lq - S) - q) * np.cos(m[:, None] + m[None, :])
        S_new = np.outer(self.u_max, self.u_max) * S_new / 2
        C = self.u_max * np.diag(np.exp(-np.diag(S) / 2) * np.cos(m))   # inv(x.cov) is already premultiplied
        return stats.multivariate_normal(m_new, S_new, allow_singular=True), C
    
    def get_action(self, x):
        # get the moment-matched preliminary control (u_)
        u_, c_ = self.predict(x, deterministic=True)
        u, c = self.squash_sin(u_) # squash u_ into [-1, 1] with the sine function (x, u)
        return u, c_ @ c


class PILCO:
    # reward should be a callable function that accepts multivariate normal random variable
    def __init__(self, state_dim, control_dim, reward, policy=None, u_max=1.0, x_init=None, horizon=20):
        self.model = MultiGPR(state_dim + control_dim, state_dim)
        if policy is None:
            self.policy = RBFPolicy(state_dim, control_dim, u_max=u_max)
        else:
            self.policy = policy
        
        self.reward = reward
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.num_episode_run = 0
        if x_init == None:
            self.x_init = stats.multivariate_normal(np.zeros(self.state_dim), np.eye(self.state_dim))
        else:
            self.x_init = x_init    # check x_init is a stats.multivariate_normal class object

    def train_model(self, X, y):
        assert X.shape[1] == self.state_dim + self.control_dim
        assert y.shape[1] == self.state_dim
        print('The model is under training...')
        tic = time.time()
        self.model.fit(X, y)
        tac = time.time()
        print('The model is trained with time %.3fs' % (tac - tic))

    def propagate(self, x):
        # Input: a state vector (x; D)
        # Output: action (u; F), a next state vector (x_new; D)
        assert self.model_trained   # check self.model is trained

        # compute the approximate joint Gaussian distribution of x and u
        print('Getting action with the current policy...')
        u, c = self.policy.get_action(x)
        m = np.hstack((x.mean, u.mean))
        S = np.block([[x.cov, c], [c.T, u.cov]])
        xu = stats.multivariate_normal(m, S, allow_singular=True)
        
        # compute the state differece with the model
        print('Calculating the successor state...')
        d_x, dc = self.model.predict(xu)
        mx_new = x.mean + d_x.mean
        Sx_new = x.cov + d_x.cov + dc[:self.state_dim, :] + dc[:self.state_dim, :].T
        x_new = stats.multivariate_normal(mx_new, Sx_new, allow_singular=True)
        return u, x_new

    def run_episode(self, x, verbose=True):
        # run an episode with the current policy
        tic = time.time()
        t = 0
        x_old = x
        state = [x_old]
        action = [None]
        reward = [0.] # ignore the initial state reward
        ep = namedtuple('episode', 'state, action, reward')  # list of tuples: (state-action pair (MVN), reward) / maybe named tuple would be better
        while t < self.horizon:
            t += 1
            if verbose:
                print('State (t=%i) is running...' %t, end='\r')
            else:
                pass
            u, x_new = self.propagate(x_old)
            r = self.reward(x_new, u)
            state.append(x_new)
            action.append(u)
            reward.append(r)    # the reward function evaluates the action and the resulting state
        tac = time.time()
        if verbose:
            print('The Episode is ended with time %.3fs and the reward %.3f' %(tac - tic, r), end='\r')
        else:
            pass
            x_old = x_new
        return ep(state, action, reward)
    
    def get_neg_reward(self, theta, verbose=False):
        # this is the objective function to be minimized
        self.num_episode_run += 1
        self.policy.update(theta)
        episode = self.run_episode(self.x_init, verbose)
        return -np.sum(episode.reward)

    def optimize_policy(self, method, n_restarts=0, display=True):
        print('The policy is under otimization...')
        tic = time.time()
        assert self.model_trained   # check self.model is trained
        self.num_episode_run = 0    # initialize the number of episode run
        self.policy_neg_reward = []
        res = minimize(self.get_neg_reward, np.random.randn(self.policy.theta_len), method=method, callback=self.op_callback, options={'maxiter': 100, 'disp': False})
        theta_new = res.x
        reward = res.fun
        if n_restarts > 0:
            theta_ls = []
            reward_ls = []
            theta_ls.append(theta_new)
            reward_ls.append(reward)
            while n_restarts > 0:
                n_restarts -= 1
                res = minimize(self.get_neg_reward, np.random.randn(self.policy.theta_len))
                theta_new = res.x
                reward = res.fun
                theta_ls.append(theta_new)
                reward_ls.append(reward)
            theta_new = theta_ls[np.argmin(reward_ls)]
            reward = reward_ls[np.argmin(reward_ls)]
        tac = time.time()
        print('The policy optimization is ended with time %.3fs and %i episodes' % (tac - tic, self.num_episode_run))
        if display:
            plt.plot(self.policy_neg_reward)
            plt.show()        
        return theta_new, reward

    def op_callback(self, theta):
        r = self.get_neg_reward(theta)
        print(r)
        self.policy_neg_reward.append(r)

    @property
    def model_trained(self):
        return hasattr(self.model, 'X_train_')