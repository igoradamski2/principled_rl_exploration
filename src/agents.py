import numpy as np

from sklearn.neighbors import KernelDensity
from copy import deepcopy
from indexedproperty import indexedproperty
import importlib
from scipy import stats

from .agent_utils import Memory
from .models import *

class Agent(object):
    
    def __init__(self, gamma, compute_Q_every_step):

        self.gamma                = gamma
        self.compute_Q_every_step = compute_Q_every_step
        
        self.memory_buffer        = []
        self.since_last_Q_compute = self.compute_Q_every_step
    
    def initialize_in_environment(self, env):
        '''
        Initialize agent in the environment
            - Get all possible state-action pairs
            - Initialize Q function dimensions
            - env is environment object
        '''
        if not hasattr(env, '_sa_pairs'):
            env.get_all_state_action_pairs()
        
        self._sa_pairs      = deepcopy(env._sa_pairs)
        self._env_states    = deepcopy(env._states)
        self._env_actions   = deepcopy(env._actions) # superset of actions

        self.initialize_R_D_models()

        # Initialize random policy
        self.pi = np.ones((len(self._env_states), len(self._env_actions)))
        self.pi = self.pi/self.pi.sum(axis = 1, keepdims=True)

    def update_memory_buffer(self, situation):
        '''
        Updates the memory buffer of the agent
        given a situation = [s, a, s_, r]
        '''

        self.memory_buffer.append(situation)
        self.update_observed_r_d(situation)

    def update_observed_r_d(self, situation):
        '''
        Update empirical dictionaries of encountered rewards and dynamics
        per each (state,action) pair
        Input: situation = [s, a, s_, r]
        '''
        if not hasattr(self, 'observed_r'):
            self.observed_r = {pair:[] for pair in self._sa_pairs}
        if not hasattr(self, 'observed_d'):
            self.observed_d = {pair:[] for pair in self._sa_pairs}
        
        self.observed_r[(situation[0], situation[1])].append(situation[3])
        self.observed_d[(situation[0], situation[1])].append(situation[2])
    
    def update(self, situation):
        '''
        Updates all agents models 
        given a situation = [s,a,s_,r]
        '''
        # Update the memory buffer
        self.update_memory_buffer(situation)

        # Update rewards and dynamics model
        self.update_R_D()
    
    def risk_aware_policy_iteration(self, maxit):
        '''
        Find Q and u given dynamics and rewards model contained in self
        '''

        num_states  = len(self._env_states)
        num_actions = len(self._env_actions)

        # Initialize policy to always pick 1st action
        pi = np.zeros((num_states, num_actions))
        pi[:,0] = 1
        pi_ = deepcopy(pi)

        for it in range(maxit):

            # First we will solve for optmial Q under policy pi, given expectation matrices
            D = self.get_moment_dynamics_matrix(which_moment = 1)
            R = self.get_moment_rewards_matrix(which_moment = 1)
            
            Q, pi_ = self.solve_for_Q(pi, D, R)

            # Now we solve the equation for the variance of return
            u = self.solve_variance_return(Q, pi_)

            # Check convergence
            if np.array_equal(pi, pi_):
                break
            
            pi = pi_
        
        return Q, u

    def solve_for_Q(self, pi, D, R):
        '''
        Solves for Q given models as dynamics and rewards
        '''
        # Find D|pi and R|pi
        D_pi = np.array([np.dot(D[s,:,:], pi[s,:]) for s in range(pi.shape[0])]) # maybe np.einsum('ijk,ik->ij', P, pi)
        R_pi = np.array([np.dot(R[s,:], pi[s,:]) for s in range(pi.shape[0])]) # maybe np.einsum('ij,ij->i', R, pi)
        
        # Solve for v and Q
        v = np.linalg.solve(np.eye(*D_pi.shape) - self.gamma*D_pi, R_pi)
        Q = R + self.gamma * np.einsum('ijk,j->ik', D, v) # equivalent to D[:,0,:]*v[0] + D[:,1,:]*v[1] + D[:,2,:]*v[2]
        
        # Update policy
        pi_  = np.zeros(pi.shape)
        idx  = np.array(list(zip(np.arange(0,len(self._env_states),1),np.argmax(Q, axis = 1))))
        
        pi_[idx[:,0], idx[:,1]] = 1

        return Q, pi_
    
    def solve_variance_return(self, Q, pi):
        '''
        Solves for the variance of return (inefficiently)
        '''
        D  = self.get_moment_dynamics_matrix(which_moment = 1)
        R  = self.get_moment_rewards_matrix(which_moment = 1)
        R2 = self.get_moment_rewards_matrix(which_moment = 2)

        ret_sq = np.zeros((len(self._env_states), len(self._env_actions)))

        converged = False
        while not converged:
            first_term = np.einsum('ijk,j', D, 2 * self.gamma * np.einsum('ij,ij,ij->i', pi, R, Q))
            sec_term   = np.einsum('ijk,j', D, (self.gamma**2) * np.einsum('ij,ij->i', pi, ret_sq))
            
            ret_sq_ = R2 + first_term + sec_term

            if np.max(np.abs(ret_sq - ret_sq_)) < 0.01:
                converged = True
            
            ret_sq = ret_sq_

        return ret_sq - Q**2

    def perform_policy_iteration(self, D, R, maxit):
        '''
        Finds optimal Q given dynamics and rewards
        '''

        num_states  = len(self._env_states)
        num_actions = len(self._env_actions)

        assert D.shape == (num_states, num_states, num_actions), "Dynamics D must be of shape (|S|, |S|, |A|)"
        assert R.shape == (num_states, num_actions), "Rewards R must be of shape (|S|, |A|)"

        # Initialize policy to always pick 1st action
        pi = np.zeros((num_states, num_actions))
        pi[:,0] = 1
        pi_ = deepcopy(pi)

        for it in range(maxit):

            # Find D|pi and R|pi
            D_pi = np.array([np.dot(D[s,:,:], pi_[s,:]) for s in range(pi_.shape[0])]) # maybe np.einsum('ijk,ik->ij', P, pi)
            R_pi = np.array([np.dot(R[s,:], pi[s,:]) for s in range(pi.shape[0])]) # maybe np.einsum('ij,ij->i', R, pi)

            # Solve for v and Q
            v = np.linalg.solve(np.eye(*D_pi.shape) - self.gamma*D_pi, R_pi)
            Q = R + self.gamma * np.einsum('ijk,j->ik', D, v) # equivalent to D[:,0,:]*v[0] + D[:,1,:]*v[1] + D[:,2,:]*v[2]

            # Update policy
            pi_  = np.zeros((num_states, num_actions))
            idx  = np.array(list(zip(np.arange(0,num_states,1),np.argmax(Q, axis = 1))))

            pi_[idx[:,0], idx[:,1]] = 1

            # Check convergence
            if np.array_equal(pi, pi_):
                break
            
            pi = pi_
        
        return Q, pi
    
    def get_monte_carlo_Q(self, num_samples):
        '''
        Perform monte carlo Q estimation
        Inputs: 
            num_samples : number of samples of matrices R and D
        
        Outputs:
            Q_mean : mean of Q 
            Q_var  : variance of Q 
        '''
        num_states  = len(self._env_states)
        num_actions = len(self._env_actions) 

        Qs = np.zeros((num_samples, num_states, num_actions))

        # Sample parameters from the posterior
        D = self.get_sampled_moments_dynamics_matrix(which_moment = 0, num_samples = num_samples)
        R = self.get_sampled_moments_rewards_matrix(which_moment = 1, num_samples = num_samples)

        for it in range(num_samples):
            Qs[it], _ = self.solve_for_Q(self.pi, D[it], R[it])

        Q_mean = np.mean(Qs, axis = 0)
        Q_var  = np.var(Qs, axis = 0)

        return Q_mean, Q_var

    def take_action(self, state):

        if self.since_last_Q_compute >= self.compute_Q_every_step:
            self.get_Q_and_u()
            self.since_last_Q_compute = 0
        
        a = self.make_decision(state)
        
        self.since_last_Q_compute += 1

        return a

    def initialize_R_D_models(self):
        '''
        Must be implemented by the child environment

        Initializes _R and _D models
        '''
        raise NotImplementedError

    def update_R_D(self):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError
    
    def get_moment_dynamics_matrix(self, *args, **kwargs):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError
    
    def get_moment_rewards_matrix(self, *args, **kwargs):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError

    def get_sampled_moments_dynamics_matrix(self, *args, **kwargs):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError

    def get_sampled_moments_rewards_matrix(self, *args, **kwargs):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError

    def get_Q_and_u(self, *args, **kwargs):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError

    def make_decision(self, *args, **kwargs):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError
      

class KDEAgent(Agent):

    def __init__(self, params):
        '''
        Initializes the KDE agent

        Inputs: params -> dict
                must have attributes:
                    'kernel_bandwidth': kernel_bandwidth for KernelDensity : float
                    'kernel': kernel for KernelDensity : string
                    'init_datapoint': initial datapoint to initialize the KDEs : float
                    'init_info_gain': initial information gain assigned to all actions : float

                    'gamma': gamma parameter for RL : float
        '''

        self.kernel_bandwidth = params['kernel_bandwidth']
        self.kernel           = params['kernel']
        self.init_datapoint   = params['init_datapoint']
        self.init_info_gain   = params['init_info_gain']

        self.logger           = Memory()

        super(KDEAgent, self).__init__(params['gamma'])
    
    def _initialize_R_D_models(self):
        '''
        Initializes the _R and _D models
        '''
        # Rewards
        self._R_ = {}
        for pair in self._sa_pairs:
            kde = KernelDensity(bandwidth=self.kernel_bandwidth, \
                                kernel=self.kernel)
            kde.fit([[self.init_datapoint]])
            self._R_[pair] = kde
        
        # Dynamics
        self._D_ = {0 for pair in self._sa_pairs}
    
    # Here we define all the properties whose changes we wish to add to the self.logger ----
    @indexedproperty
    def R_(self, key):
        return self._R_[key]
    
    @indexedproperty
    def D_(self):
        return self._D_
    
    @R_.setter
    def R_(self, key, value):
        self._on_change('R_', key, value)
        self._R_[key] = value
    
    @D_.setter
    def D_(self, key, value):
        self._on_change('D_', key, value)
        self._D_[key] = value
    
    def _on_change(self, which, key, value):

        self.logger.update(which, key, value)
    
    # -------------------------------------------------------------------------

    def _update_R(self):
        '''
        Updates the reward distribution according to the latest self.observed_r
        Returns: R_ the distribution of rewards model
        '''

        if not hasattr(self, 'R_'):
            NotImplementedError("Agent not initialized in environment")
        else:
            for pair in self._sa_pairs:
                if self.observed_r[pair]:
                    kde = KernelDensity(bandwidth=self.kernel_bandwidth, \
                                        kernel=self.kernel)
                    kde.fit(np.array(self.observed_r[pair]).reshape(-1,1))
                    self.R_[pair] = kde
        
        # This fires the setter (logging)
        self.R_ = self.R_
    
    def _update_D(self):
        '''
        Updates the dynamics distribution according to the latest self.observed_d
        Returns: D_ the distribution of rewards model
        '''
        if not hasattr(self, 'D_'):
            NotImplementedError("Agent not initialized in environment")
    
    def sample_R_and_D_matrices(self, num_samples):
        '''
        Samples num_samples matrices for the reward and distribution

        Inputs: num_samples : int
                R_ : dictionary of KDE models (dict)
                D_ : dictionary of dynamics (dict)

        Return: R : np.array of dimension (num_samples, |S|, |A|)
                D : np.array of dimension (num_samples, |S|, |S|, |A|)
        '''
        # First sample R matrix
        curr_R_ = deepcopy(self.R_)

        R_matrix = np.zeros((num_samples, len(self._env_states), len(self._env_actions)))
        for pair in curr_R_.keys():
            kde = curr_R_[pair]
            R_matrix[:, pair[0], pair[1]] = kde.sample(num_samples).reshape(-1)
        
        # Second sample D matrix
        D_matrix = np.ones((num_samples, len(self._env_states), len(self._env_states), len(self._env_actions)))

        return R_matrix, D_matrix


    @classmethod
    def default(cls):
        params = {'kernel_bandwidth': 1, 
                  'kernel': 'gaussian',
                  'init_datapoint': 0,
                  'init_info_gain': 100,
                  'gamma': 0.8, 
        }
        return cls(params)

class BayesianAgent(Agent):
    
    def __init__(self, params):
        '''
        Initializes the Bayesian Agent (agent keeps bayesian models of world).
        Input: params -> dict
                params['reward_model'] -> Class Name (from models.py) of model for rewards
                params['dyna_model'] -> Class Name (from models.py) of model for dynamics
                params['gamma'] -> gamma

                Optional:
                params['reward_params'] -> dict (for reward model to initialize)
                params['dyna_params'] -> dict (for dynamics model to initialize)
        '''
        self.reward_model = params['reward_model']
        self.dyna_model   = params['dyna_model']

        # Initialize Q and u estimation methods
        self.Q_and_u_method         = params['Q_and_u_method']
        self.Q_and_u_method_params  = params['Q_and_u_method_params']

        # Initialize decision making methods
        self.decision_making_method = params['decision_making_method']

        if 'reward_params' in params.keys():
            self.reward_params = params['reward_params']
        
        if 'dyna_params' in params.keys():
            self.dyna_params = params['dyna_params']
        
        if 'logger' in params.keys():
            self.logger = Memory()
    
        super(BayesianAgent, self).__init__(params['gamma'], params['compute_Q_every_step'])
        
    def initialize_R_D_models(self):
        '''
        Initializes R and D models.
        Agent must be first initialized in environment.
        '''
        self._R_ = {}
        self._D_ = {}

        reward_class = getattr(importlib.import_module(".models", package='src'), self.reward_model)
        dyna_class = getattr(importlib.import_module(".models", package='src'), self.dyna_model)

        for pair in self._sa_pairs:
            # Rewards
            if hasattr(self, 'reward_params'):
                self.R_[pair] = reward_class(self.reward_params)
            else:
                self.R_[pair] = reward_class.default()

            # Dynamics
            if hasattr(self, 'dyna_params'):
                self.D_[pair] = dyna_class(self.dyna_params)
            else:
                self.D_[pair] = dyna_class.default(len(self._env_states))
        
    @indexedproperty
    def R_(self, key):
        return self._R_[key]
    
    @indexedproperty
    def D_(self, key):
        return self._D_[key]
    
    @R_.setter
    def R_(self, key, value):
        self._on_change('R_', key, value)
        self._R_[key] = value
    
    @D_.setter
    def D_(self, key, value):
        self._on_change('D_', key, value)
        self._D_[key] = value
    
    def _on_change(self, which, key, value):
        print('change!')
        if hasattr(self, 'logger'):
            self.logger.update(which, key, value)
        else:
            pass
    
    def update_R_D(self):
        '''
        Updates the reward distribution according to the latest self.observed_r
        Returns: R_ the distribution of rewards model
        '''

        if not hasattr(self, '_R_') or not hasattr(self, '_D_'):
            NotImplementedError("Agent not initialized in environment")
        else:
            for pair in self._sa_pairs:
                if self.observed_r[pair]:
                    self.R_[pair].update(self.observed_r[pair])
                if self.observed_d[pair]:
                    self.D_[pair].update(self.observed_d[pair])

    def sample_R_and_D_matrices(self, num_samples):
        '''
        Samples num_samples matrices for the reward and distribution

        Inputs: num_samples : int
                R_ : dictionary of reward models (dict)
                D_ : dictionary of dynamics models (dict)

        Return: E[R] : np.array of dimension (num_samples, |S|, |A|)
                D    : np.array of dimension (num_samples, |S|, |S|, |A|)
        '''
        # First sample E[R] matrix
        curr_R_ = deepcopy(self.R_)

        R_matrix = np.zeros((num_samples, len(self._env_states), len(self._env_actions)))
        for pair in curr_R_.keys():
            model = curr_R_[pair]
            R_matrix[:, pair[0], pair[1]] = 1
        
        # Second sample D matrix
        D_matrix = np.ones((num_samples, len(self._env_states), len(self._env_states), len(self._env_actions)))

        return R_matrix, D_matrix
    
    def get_moment_rewards_matrix(self, which_moment):
        '''
        Gets the expected reward matrix, as the which_moment'th moment w.r.t. model parameters

        Inputs: which_moment = int

        Return: E_{theta}[E[R^which_moment]] : np.array of dimension (|S|, |A|)
        '''
        
        R_matrix = np.zeros((len(self._env_states), len(self._env_actions)))
        for pair in self._R_.keys():
            R_matrix[pair[0], pair[1]] = self.R_[pair].get_predictive_moment(which_moment)
        
        return R_matrix
    
    def get_sampled_moments_rewards_matrix(self, which_moment, num_samples):
        '''
        Gets the reward matrix for a sampled set of parameters, 
        as the which_moment'th moment w.r.t. model parameters

        Inputs: which_moment = int
                num_samples = int

        Return: num_samples of E[R^which_moment|theta] : np.array of dimension (num_samples, |S|, |A|)
        '''

        R_matrix = np.zeros((num_samples, len(self._env_states), len(self._env_actions)))
        for pair in self._R_.keys():
            R_matrix[:, pair[0], pair[1]] = self.R_[pair].get_sampled_moments(which_moment, num_samples)
        
        return R_matrix

    def get_moment_dynamics_matrix(self, which_moment):
        '''
        Gets the dynamics matrix, as the which_moment'th moment w.r.t. model parameters

        Inputs: which_moment = int

        Return: E_{theta}[E[D^which_moment]] : np.array of dimension (|S|, |S|, |A|) (s, s', a)
        '''

        D_matrix = np.zeros((len(self._env_states), len(self._env_states), len(self._env_actions)))
        for pair in self._D_.keys():
            D_matrix[pair[0], :, pair[1]] = self.D_[pair].get_predictive_moment(which_moment)

        return D_matrix

    def get_sampled_moments_dynamics_matrix(self, which_moment, num_samples):
        '''
        Gets the dynamics matrix for a sampled set of parameters, 
        as the which_moment'th moment w.r.t. model parameters

        Inputs: which_moment = int
                num_samples = int

        Return: num_samples of E[D^which_moment|theta] : np.array of dimension (num_samples, |S|, |S|, |A|) (s, s', a)
        '''

        D_matrix = np.zeros((num_samples, len(self._env_states), len(self._env_states), len(self._env_actions)))
        for pair in self._D_.keys():
            D_matrix[:, pair[0], :, pair[1]] = self.D_[pair].get_sampled_moments(which_moment, num_samples)

        return D_matrix
    
    def get_Q_and_u(self):
        '''
        Populates the agent class with Q and u estimations
        '''
        if self.Q_and_u_method == 'monte_carlo':
            self.Q, self.u = self.get_monte_carlo_Q(self.Q_and_u_method_params)
        
    def make_decision(self, state):
        '''
        Makes a decision in the current state, given decision making method
        '''
        if self.decision_making_method == 'SOSS':
            # Get the list of actions in this state
            actions = np.array([pair[1] for pair in self._sa_pairs if pair[0] == state])
            p_a     = np.ones(len(actions)) 
            for a in actions:
                for a_ in actions:
                    if a == a_:
                        continue
                    else:
                        p_a[a] *= stats.norm.cdf((self.Q[state,a] - self.Q[state,a_])/np.sqrt((self.u[state,a] + self.u[state,a_])))
            
            p_a = p_a/np.sum(p_a, keepdims=True)

            # Update current policy
            self.pi[state,:] = deepcopy(p_a)

            # Sample from policy
            action = np.random.choice(a=actions, size=1, p=p_a)
        
        return int(action)

    @classmethod
    def default(cls):
        params = {
            'reward_model': 'BayesianGaussian',
            'dyna_model': 'BayesianCategorical',
            'gamma': 0.9,
            'Q_and_u_method': 'monte_carlo',
            'Q_and_u_method_params': 20,
            'decision_making_method': 'SOSS', # this stands for Stochastic Optimal Strategy Search
            'compute_Q_every_step': 1,
        }
        return cls(params)
        
    
