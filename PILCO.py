import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
import scipy.stats as stats
from scipy.linalg import solve_triangular, solve
from collections import namedtuple
from scipy.optimize import minimize
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
            self.gps[i] = GaussianProcessRegressor(kernel=kernel, **kwargs)
        
    def fit(self, X, y):
        assert X.shape[1] == self.input_dim
        assert y.shape[1] == self.output_dim

        self.X_train_ = X.copy()    # (N, D)    
        self.y_train_ = y.copy()    # (N, F)

        self.length = np.zeros((self.output_dim, self.input_dim))   # (F, D)
        self.const = np.ones(self.output_dim)  # (F)
        for i in range(self.output_dim):
            self.gps[i].fit(X, y[:, i])
            for j in range(len(self.gps[0].kernel_.hyperparameters)):
                if 'length_scale' in self.gps[i].kernel_.hyperparameters[j][0]:
                    self.length[i] = self.gps[i].kernel_.get_params()[self.gps[i].kernel_.hyperparameters[j][0]]
                elif 'constant_value' in self.gps[i].kernel_.hyperparameters[j][0]:
                    self.const[i] = self.gps[i].kernel_.get_params()[self.gps[i].kernel_.hyperparameters[j][0]]
                else:
                    pass
    
    def predict(self, x, deterministic=False):
        # Input: scipy.stats.mltivariate_normal class object (x)
        # Output: scipy.stats.mltivariate_normal class object (m_new, S_new), input-output covariance matrix (C)
        m = x.mean  # (D)
        S = x.cov   # (D, D)
        m_new = np.zeros(self.output_dim)   # (F)
        S_new = np.zeros((self.output_dim, self.output_dim))    # (F, F)
        C = np.zeros((self.input_dim, self.output_dim))   # (D, F)

        qs = np.zeros((self.output_dim, self.X_train_.shape[0]))    # (F, N)
        nu = self.X_train_ - m[None, :]    # (N, D)
        for i in range(self.output_dim):
            TI = np.linalg.inv(S + np.diag(self.length[i]))   # (D, D)
            qs[i] = self.const[i] / np.sqrt(np.linalg.det(S @ np.diag(1 / self.length[i]) + np.eye(self.input_dim))) \
                * np.diag(np.exp(-nu @ TI @ nu.T / 2))   # take log?
            m_new[i] = self.gps[i].alpha_.dot(qs[i])
            C[:, i] = np.dot(S @ TI @ nu.T , qs[i] * self.gps[i].alpha_)

        for i in range(self.output_dim):
            for j in range(i, self.output_dim):
                R = S @ (np.diag(1 / self.length[i]) + np.diag(1 / self.length[j])) + np.eye(self.input_dim)    # (D, D)
                Zi = nu @ np.diag(1 / self.length[i]) # (N, D)
                Zj = nu @ np.diag(1 / self.length[j]) # (N, D)
                # the simplified version of computing Q from Eq. (2.53) in Deisenroth (2010)
                ln_Q = np.log(self.const[i]) + np.log(self.const[j]) \
                    - (smaha(nu, Q=np.diag(1 / self.length[i])) + smaha(nu, Q=np.diag(1 / self.length[j])).T - smaha(Zi, -Zj, solve(R, S))) / 2 # (N, N)
                Q = np.exp(ln_Q)
                Q /= np.sqrt(np.linalg.det(R))
                S_new[i, j] = self.gps[i].alpha_.dot(Q @ self.gps[j].alpha_) - m_new[i] * m_new[j]
                S_new[j, i] = S_new[i, j]
                if not deterministic:
                    if j == i:
                        S_new[i, j] += self.const[i] - np.trace(solve_triangular(self.gps[i].L_.T, solve_triangular(self.gps[i].L_, Q, lower=True), lower=False))
                else:
                    pass    # in the case of deterministic (e.g., RBFPolicy), the model uncertainty is not considered
        else:
            return stats.multivariate_normal(m_new, S_new), C


class RBFPolicy(MultiGPR):
    def __init__(self, state_dim, control_dim, level=30, u_max=1.0, kernel=None, **kwargs):
        if kernel is None:
            kernel = C(1.0, 'fixed') * RBF(np.exp(np.random.randn(state_dim)), 'fixed') # use a fixed kernel
        else:
            kernel = kernel
        super().__init__(state_dim, control_dim, kernel=kernel, alpha=1e-2, **kwargs)
        self.level = level
        self.u_max = u_max

        # initialize the policy with random parameters
        self.theta_dims = np.array([[self.level, self.input_dim], [self.level, self.output_dim], [self.output_dim, self.input_dim]])  # (M, T, legnth_scale)
        self.theta_len = np.multiply(*self.theta_dims.T).sum()
        self.M = np.random.randn(*self.theta_dims[0])
        self.T = np.random.randn(*self.theta_dims[1])
        self.fit(self.M, self.T)
        
    def randomize(self):
        print('randomizing the policy')
        theta = np.random.randn(self.theta_len)
        self.update(theta)
    
    def update(self, theta):
        # theta is a vector, in which length_scale values are taken logarithm
        self.M, self.T, self.Lambda = np.split(theta, np.cumsum(np.multiply(*self.theta_dims.T))[:-1])
        self.M = self.M.reshape(*self.theta_dims[0])
        self.T = self.T.reshape(*self.theta_dims[1])
        self.Lambda = np.exp(self.Lambda.reshape(*self.theta_dims[2]))
        for i in range(self.output_dim):
            self.gps[i].kernel_.set_params(k2__length_scale=self.Lambda[i])
        self.fit(self.M, self.T)
    
    def squash_sin(self, x):
        # a sine function squashing restricts the preliminary policy output to [-1, 1]
        # need to check: why we only consider the diagonals of S? (the current code is from 'nrontsis' and the Matlab code)
        m = x.mean
        S = x.cov

        m_new = self.u_max * np.exp(-np.diag(S) / 2) * np.sin(m)    # 
        lq = -(np.diag(S)[:, None] + np.diag(S)[None, :]) / 2
        q = np.exp(lq)
        S_new = (np.exp(lq + S) - q) * np.cos(m[:, None] - m[None, :]) - (np.exp(lq - S) - q) * np.cos(m[:, None] + m[None, :])
        S_new = np.outer(self.u_max, self.u_max) * S_new / 2
        C = self.u_max * np.diag(np.exp(-np.diag(S) / 2) * np.cos(m))   # already inv(x.cov) is premultiplied
        return stats.multivariate_normal(m_new, S_new), C
    
    def get_action(self, x):
        # get the moment-matched preliminary control (u_)
        u_, c_ = self.predict(x, deterministic=True)
        u, c = self.squash_sin(u_) # squash u_ into [-1, 1] with the sine function (x, u)
        return u, c_ @ c


class PILCO:
    # reward should be a callable function that accepts multivariate normal random variable
    def __init__(self, state_dim, control_dim, reward, policy=None, x_init=None, horizon=20):
        self.model = MultiGPR(state_dim + control_dim, state_dim)
        if policy is None:
            self.policy = RBFPolicy(state_dim, control_dim)
        else:
            self.policy = policy
        
        self.reward = reward
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        if x_init == None:
            self.x_init = stats.multivariate_normal(np.zeros(self.state_dim), np.eye(self.state_dim))
        else:
            self.x_init = x_init    # check x_init is a stats.multivariate_normal class object

    def train_model(self, X, y):
        assert X.shape[1] == self.state_dim + self.control_dim
        assert y.shape[1] == self.state_dim
        self.model.fit(X, y)

    def propagate(self, x):
        assert self.model_trained   # check self.model is trained
        # Input: a state vector (x; D)
        # Output: action (u; F), a next state vector (x_new; D)

        # compute the approximate joint Gaussian distribution of x and u
        u, c = self.policy.get_action(x)
        m = np.hstack((x.mean, u.mean))
        S = np.block([[x.cov, c], [c.T, u.cov]])
        xu = stats.multivariate_normal(m, S)
        
        # compute the state differece with the model
        d_x, dc = self.model.predict(xu)
        mx_new = x.mean + d_x.mean
        Sx_new = x.cov + d_x.cov + dc[:self.state_dim, :] + dc[:self.state_dim, :].T
        x_new = stats.multivariate_normal(mx_new, Sx_new)
        return xu, x_new

    def run_episode(self, x):
        t = 0
        trace = []
        reward = []
        ep = namedtuple('episode', 'trace, reward')  # list of tuples: (state-action pair (MVN), reward) / maybe named tuple would be better
        x_old = x
        while t < self.horizon:
            xu, x_new = self.propagate(x_old)
            trace.append(xu)
            reward.append(self.reward(x_old))
            x_old = x_new
            t += 1
        return ep(trace, reward)
    
    def get_neg_reward(self, theta):
        self.policy.update(theta)
        episode = self.run_episode(self.x_init)
        return -np.sum(episode.reward)

    def optimize_policy(self, n_restarts=0):
        assert self.model_trained   # check self.model is trained
        res = minimize(self.get_neg_reward, np.random.randn(self.policy.theta_len))
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
        return theta_new, reward

    @property
    def model_trained(self):
        return hasattr(self.model, 'X_train_')