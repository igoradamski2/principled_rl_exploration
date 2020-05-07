import numpy as np

from sklearn.neighbors import KernelDensity
from copy import deepcopy

class Agent(object):
    
    def __init__(self, gamma):

        self.gamma = gamma
    
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
        self._env_actions   = deepcopy(env._actions)
        self._memory_buffer = []

        self._initialize_R_D_models()

    def update_memory_buffer(self, situation):
        '''
        Updates the memory buffer of the agent
        given a situation = [s, a, s_, r]
        '''
        assert hasattr(self, '_memory_buffer'), "Agent not initialized in the environment"

        self._memory_buffer.append(situation)
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

        # Update rewards model
        self._update_R()

        # Update dynamics model
        self._update_D()

    def perform_policy_iteration(self, D, R, maxit):
        '''
        Finds optimal Q given dynamics and rewards
        '''
        

    def take_action(self, state):
        return np.random.choice(3)

    def _initialize_R_D_models(self):
        '''
        Must be implemented by the child environment

        Initializes _R and _D models
        '''
        raise NotImplementedError

    def sample_R_and_D_matrices(self, num_samples):
        '''
        Must be implemented by the child environment

        Samples num_samples matrices for the reward and distribution
        '''
        raise NotImplementedError

    def _update_R(self):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError

    def _update_D(self):
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

        self.logger           = {}

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
    @property
    def R_(self):
        return self._R_
    
    @property
    def D_(self):
        return self._D_
    
    @R_.setter
    def R_(self, value):
        self._on_change('R_')
        self._R_ = value
    
    @D_.setter
    def D_(self, value):
        self._on_change('D_')
        self._D_ = value
    
    def _on_change(self, which):
        if not which in self.logger.keys():
            self.logger[which] = [getattr(self, which)]
        else:
            self.logger[which].append(getattr(self, which))
    
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
        #self.R_ = self.R_
    
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
    # This agent will update the posteriors starting from some priors
    pass