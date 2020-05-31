import numpy as np
from copy import deepcopy

from .distributions import Distribution


class Environment(object):

    def __init__(self):
        
        # Get dynamics and rewards
        self.D = self.get_dynamics()
        self.R = self.get_rewards()

        # Get initial state
        self.s = self.get_initial_state()

        # Set time to 0
        self.t = 0
    
    def step(self, a):
        '''
        Perform a step in the environment
            - given an action return a reward
            - set current state to new state    
        Current state in the environment is the self.s variable
        '''
        # Get new state
        s_ = self.D(self.s, a)

        # Get reward
        r = self.R(self.s, a, s_)

        # Set new state
        self.s = s_

        # Evolve time
        self.t += 1

        return r
    
    def reset(self):
        '''
        Reset the environment state and time to the initial state
        '''
        self.s = self.get_initial_state()
        self.t = 0
    
    def play_episode(self, num_steps, agent):
        '''
        Propagate a episode of num_steps steps, 
        starting from the current state
        given an agent 

        Returns: trained agent
        '''
        self.reset()

        # Initialize agent in the environment
        agent.initialize_in_environment(self)

        for step in range(num_steps):
            
            # Take an action
            a = agent.take_action(self.s)

            # Observe new state and reward
            s  = deepcopy(self.s) 
            r  = self.step(a)
            s_ = deepcopy(self.s)

            # Update agent
            agent.update([s, a, s_, r])

            print('I chose action {}'.format(a))
        
        return agent

    def get_dynamics(self):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError
    
    def get_rewards(self):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError

    def get_initial_state(self):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError

    def get_possible_actions(self):
        '''
        Returns possible actions in the current state of environment
        '''
        if not hasattr(self, '_sa_pairs'):
            self.get_all_state_action_pairs()
        
        return np.array([pair[1] for pair in self._sa_pairs if pair[0] == self.s])

    def get_all_state_action_pairs(self):
        '''
        Must be implemented by the child environment

        Returns all possible state-action pairs
            - populate self._states with all state numbers (np.array)
            - populate self._actions with all action numbers (np.array)
            - populate self._sa_pairs with all possible state-action pairs (list of tuples)
        '''
        raise NotImplementedError


class MultiArmedBandit(Environment):

    def __init__(self, params):
        '''
        Initializes the Multi Armed Bandit environment

        Inputs: params -> dict
                must have attributes:
                    'num_bandits': number of bandits : int
                    'reward_distrib': distributions of bandits : list
                    'reward_distrib_params': distributions parameters : list of lists
        '''

        self.num_bandits           = params['num_bandits']
        self.reward_distrib        = params['reward_distrib'] 
        self.reward_distrib_params = params['reward_distrib_params']

        if self.num_bandits > len(self.reward_distrib) and len(self.reward_distrib) == 1:
            self.reward_distrib *= self.num_bandits

        assert len(self.reward_distrib) == self.num_bandits, "Too few distributions specified for that number of bandits"

        # Mount an environment object
        super(MultiArmedBandit, self).__init__()
         
    def get_dynamics(self):
        '''
        Returns a function D:(s,a) -> next_state
        '''
        def D(s, a):
            return s
        
        return D

    def get_rewards(self):
        '''
        Returns a function R:(s,a,s_) -> reward
        '''
        def R(s, a, s_):
            dist = getattr(Distribution, self.reward_distrib[a])
            return dist(*self.reward_distrib_params[a])
        
        return R
    
    def get_initial_state(self):
        return 0
    
    def get_all_state_action_pairs(self):
        '''
        Returns all possible state-action pairs
            - populate self._states with all state numbers (np.array)
            - populate self._actions with all action numbers (np.array)
            - populate self._sa_pairs with all possible state-action pairs (list of tuples)
        '''
        self._states   = np.array([0])
        self._actions  = np.arange(0, self.num_bandits, 1)
        self._sa_pairs = [(0,a) for a in self._actions]
    
    @classmethod
    def default(cls):
        params = {'num_bandits': 4, 
                  'reward_distrib': ['normal'], 
                  'reward_distrib_params': [[1,1], [2,2], [3,0.5], [4,5]]
        }
        return cls(params)

