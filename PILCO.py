import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
import scipy.stats as stats
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular, solve
from collections import namedtuple
from scipy.optimize import minimize


class MultiGPR:
    # Conditionally independent mlti-output Gaussian Process Regression from Deisenroth et al. (2017)
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
    
    def predict(self, x, joint=False, deterministic=False):
        # Input: scipy.stats.mltivariate_normal class object
        # Output: scipy.stats.mltivariate_normal class object, input-output covariance matrix (C)
        m = x.mean  # (D)
        S = x.cov   # (D, D)
        m_new = np.zeros(self.output_dim)   # (F)
        S_new = np.zeros((self.output_dim, self.output_dim))    # (F, F)
        C = np.zeros((self.input_dim, self.output_dim))   # (D, F)

        qs = np.zeros((self.output_dim, self.X_train_.shape[0]))    # (F, N)
        nu = self.X_train_.T - m[:, None]    # (D, N)
        for i in range(self.output_dim):
            Ti = np.linalg.inv(S + np.diag(self.length[i]))   # (D, D)
            qs[i] = self.const[i] / np.sqrt(np.linalg.det(S @ np.diag(1 / self.length[i]) + np.eye(self.input_dim))) * np.diag(np.exp(-nu.T @ Ti @ nu / 2))
            m_new[i] = self.gps[i].alpha_.dot(qs[i])
            C[:, i] = np.dot(S @ Ti @ nu , qs[i] * self.gps[i].alpha_)

        # covariance matrix requires m_new (and it is PSD)
        for i in range(self.output_dim):
            for j in range(i, self.output_dim):
                R = S @ (np.diag(1 / self.length[i]) + np.diag(1 / self.length[j])) + np.eye(self.input_dim)    # (D, D)
                Zi = nu.T @ np.diag(1 / self.length[i]) # (N, D)
                Zj = nu.T @ np.diag(1 / self.length[j]) # (N, D)
                ln_Q = np.log(np.outer(self.gps[i].kernel_(self.X_train_, m[None, :]), self.gps[j].kernel_(self.X_train_, m[None, :]))) - np.log(np.linalg.det(R)) / 2 \
                    + np.square(cdist(Zi, -Zj, 'mahalanobis', VI=solve(R, S))) / 2
                Q = np.exp(ln_Q)
                # Q = np.outer(self.gps[i].kernel_(self.X_train_, m[None, :]), self.gps[j].kernel_(self.X_train_, m[None, :])) / np.sqrt(np.linalg.det(R))\
                    #  * np.exp(np.square(cdist(Zi, -Zj, 'mahalanobis', VI=solve(R, S))) / 2)   # (N, N)
                S_new[i, j] = self.gps[i].alpha_.dot(Q @ self.gps[j].alpha_) - m_new[i] * m_new[j]
                S_new[j, i] = S_new[i, j]
                if not deterministic:
                    if j == i:
                        S_new[i, j] += self.const[i] - np.trace(solve_triangular(self.gps[i].L_.T, solve_triangular(self.gps[i].L_, Q, lower=True), lower=False))
                else:
                    pass
        if joint:
            m_new = np.hstack((m_new, m))
            S_new = np.block([[S, C], [C.T, S_new]])
            return stats.multivariate_normal(m_new, S_new), C
        else:
            return stats.multivariate_normal(m_new, S_new), C


class RBFPolicy(MultiGPR):
    def __init__(self, state_dim, control_dim, level=30, u_max=1.0, kernel=None, **kwargs):
        if kernel is None:
            kernel = C(1.0, 'fixed') * RBF(np.exp(np.random.randn(state_dim)), 'fixed')
        else:
            kernel = kernel
        super().__init__(state_dim, control_dim, kernel=kernel, n_restarts=0, alpha=1e-2, **kwargs)
        self.level = level
        self.u_max = u_max

        # initialize the policy with random parameters
        self.param_dims = np.array([[self.level, self.input_dim], [self.level, self.output_dim], [self.output_dim, self.input_dim]])  # (M, T, legnth_scale)
        self.param_len = np.multiply(*self.param_dims.T).sum()
        self.M = np.random.randn(*self.param_dims[0])
        self.T = np.random.randn(*self.param_dims[1])
        self.fit(self.M, self.T)
        
    def randomize(self):
        print('randomizing the policy')
        self.M = np.random.randn(*self.param_dims[0])
        self.T = np.random.randn(*self.param_dims[1])
        self.Lambda = np.exp(np.random.randn(*self.param_dims[2]))
        for i in range(self.output_dim):
            self.gps[i].kernel_.set_params(k2__length_scale=self.Lambda[i])
        self.fit(self.M, self.T)
    
    def update(self, theta):
        # theta is vector
        self.M, self.T, self.Lambda = np.split(theta, np.cumsum(np.multiply(*self.param_dims.T))[:-1])
        self.M = self.M.reshape(*self.param_dims[0])
        self.T = self.T.reshape(*self.param_dims[1])
        self.Lambda = np.exp(self.Lambda.reshape(*self.param_dims[2]))
        for i in range(self.output_dim):
            self.gps[i].kernel_.set_params(k2__length_scale=self.Lambda[i])
        self.fit(self.M, self.T)
    
    def squash_sin(self, x):
        m = x.mean
        S = x.cov

        m_new = self.u_max * np.exp(-np.diag(S) / 2) * np.sin(m)
        lq = -(np.diag(S)[:, None] + np.diag(S)[None, :]) / 2
        q = np.exp(lq)
        S_new = (np.exp(lq + S) - q) * np.cos(m[:, None] - m[None, :]) - (np.exp(lq - S) - q) * np.cos(m[:, None] + m[None, :])
        S_new = np.outer(self.u_max, self.u_max) * S_new / 2
        C = self.u_max * np.diag(np.exp(-np.diag(S) / 2) * np.cos(m))
        return stats.multivariate_normal(m_new, S_new), C
    
    def get_action(self, x):
        u, c = self.predict(x, deterministic=True)
        # automatically involve a squash function
        u, c2 = self.squash_sin(u)
        return u, c @ c2


class PILCO:
    def __init__(self, state_dim, control_dim, policy=None, reward=None, x_init=None, horizon=20):
        self.model = MultiGPR(state_dim + control_dim, state_dim)
        if policy is None:
            self.policy = RBFPolicy(state_dim, control_dim)
        else:
            self.policy = policy
        
        # assert reward is callable
        self.reward = reward
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        if x_init == None:
            self.x_init = stats.multivariate_normal(np.zeros(self.state_dim), np.eye(self.state_dim))
        else:
            self.x_init

    def train_model(self, X, y):
        assert X.shape[1] == self.state_dim + self.control_dim
        assert y.shape[1] == self.state_dim
        self.model.fit(X, y)

    def optimize_policy(self, n_restarts=0):

        res = minimize(self.get_reward, np.random.randn(self.policy.param_len))
        theta_new = res.x
        reward = res.fun
        if n_restarts > 0:
            theta_ls = []
            reward_ls = []
            theta_ls.append(theta_new)
            reward_ls.append(reward)
            while n_restarts > 0:
                n_restarts -= 1
                res = minimize(self.get_reward, np.random.randn(self.policy.param_len))
                theta_new = res.x
                reward = res.fun
                theta_ls.append(theta_new)
                reward_ls.append(reward)
            theta_new = theta_ls[np.argmin(reward_ls)]
            reward = reward_ls[np.argmin(reward_ls)]
        return theta_new, reward

    def get_reward(self, theta):
        self.policy.update(theta)
        episode = self.run_episode(self.x_init)
        return episode.reward.sum()

    def run_episode(self, x):
        # check self.model is trained before getting a trace
        # equivalent function of predict of PILCO
        T = self.horizon
        trace = []
        reward = []
        ep = namedtuple('episode', 'trace, reward')  # list of tuples: (state-action pair (MVN), reward) / maybe named tuple would be better
        x_old = x
        for i in range(T):
            print(i)
            xu, x_new = self.propagate(x_old)
            trace.append(xu)
            reward.append(self.reward(x_old))
            x_old = x_new        
        return ep(trace, reward)

    def propagate(self, x):
        # Input: a state vector (x; D)
        # Output: action (u; F), a next state vector (x_new; D)

        # compute the approximate joint Gaussian distribution of x and u
        u, c = self.policy.get_action(x)
        m = np.hstack((x.mean, u.mean))
        S = np.block([[x.cov, x.cov @ c], [(x.cov @ c).T, u.cov]])
        xu = stats.multivariate_normal(m, S)
        
        # compute the state differece with the model
        d_x, dc = self.model.predict(xu)
        mx_new = x.mean + d_x.mean
        Sx_new = x.cov + d_x.cov + dc[:self.state_dim, :] + dc[:self.state_dim, :].T
        x_new = stats.multivariate_normal(mx_new, Sx_new)
        return xu, x_new