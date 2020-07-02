import numpy as np
from copy import deepcopy
import random
import itertools
from tqdm import tqdm
from sys import stdout

from .distributions import Distribution


class Environment(object):

    def __init__(self):
        
        # Get dynamics and rewards
        self.D = self.get_dynamics()
        self.R = self.get_rewards()

        # Get initial state
        self.s = self.get_initial_state()

        # Get all state action pairs
        self.get_all_state_action_pairs()

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
                 cumulative regret
        '''
        self.reset()
        
        # Initialize agent in environment
        agent.initialize_in_environment(self)

        # Compute optimal behaviour in environment
        self.get_true_Q(agent.gamma)

        # Initialize regret
        regret = np.zeros(num_steps)

        # Initialize % of times selecting best action
        best_action = np.zeros(num_steps)

        for step in tqdm(range(num_steps)):
            
            # Take an action
            a = agent.take_action(self.s)

            # Observe new state and reward
            s  = deepcopy(self.s) 
            r  = self.step(a)
            s_ = deepcopy(self.s)

            # Update agent
            agent.update([s, a, s_, r])

            #print('I chose action {} in state {}'.format(a, s))
            #stdout.write('\nI chose action {} in state {}'.format(a, s))
            #stdout.flush()

            # Calculate regret
            regret[step] = (self.oracle_agent() - agent.get_cumulative_reward())

            # Calculate % of times selecting best action
            best_action[step] = 1 if a == int(np.where(self.optimal_pi[s, :] == 1)[0]) else 0
        
        agent.regret      = regret
        agent.best_action = np.cumsum(best_action)/(np.arange(len(best_action))+1)

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

    def oracle_agent(self):
        '''
        Must be implemented by the child environment

        Returns the cumulative reward of the oracle agent
        at time self.t
        '''
        raise NotImplementedError

    def get_true_Q(self, gamma):
        '''
        Populates the env with true Q value for a particular gamma
        Solves for Q given models as dynamics and rewards

        D in shape |S| |S'| |A|
        '''

        D = self.get_true_dynamics_matrix()
        R = self.get_true_rewards_matrix()

        pi = np.ones((len(self._states), len(self._actions)))
        pi = pi/pi.sum(axis = 1, keepdims=True)

        converged = False

        while not converged:

            # Find D|pi and R|pi
            D_pi = np.array([np.dot(D[s,:,:], pi[s,:]) for s in range(pi.shape[0])]) # maybe np.einsum('ijk,ik->ij', P, pi)
            R_pi = np.array([np.dot(R[s,:], pi[s,:]) for s in range(pi.shape[0])]) # maybe np.einsum('ij,ij->i', R, pi)

            # Solve for v and Q
            v = np.linalg.solve(np.eye(*D_pi.shape) - gamma*D_pi, R_pi)
            Q = R + gamma * np.einsum('ijk,j->ik', D, v) # equivalent to D[:,0,:]*v[0] + D[:,1,:]*v[1] + D[:,2,:]*v[2]

            # Update policy
            pi_  = np.zeros(pi.shape)
            idx  = np.array(list(zip(np.arange(0,len(self._states),1),np.argmax(Q, axis = 1))))

            pi_[idx[:,0], idx[:,1]] = 1

            if np.all(pi == pi_):
                converged = True
            
            pi = pi_

        self.optimal_Q  = Q
        self.optimal_pi = pi

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
    
    def oracle_agent(self):
        '''
        Returns the cumulative reward of the oracle agent
        at time self.t
        '''
        # Figure out which bandit is the best
        idx = np.unravel_index(np.argmax(self.reward_distrib_params), 
                               np.array(self.reward_distrib_params).shape)


        return self.t * self.reward_distrib_params[idx[0]][idx[1]]
    
    def get_true_dynamics_matrix(self):
        '''
        Returns the true dynamics matrix
        '''
        D = np.ones((1,1,self.num_bandits))
        return D
    
    def get_true_rewards_matrix(self):
        '''
        Returns the true rewards matrix
        '''
        R = np.array(self.reward_distrib_params)[:,0]
        R = R.reshape(len(self._states), len(self._actions))
        return R
            
    @classmethod
    def default(cls):
        params = {'num_bandits': 10, 
                  'reward_distrib': ['normal'], 
                  'reward_distrib_params': [[1,1], [2,2], [3,0.5], [4,5], [8,2], [3,1], [10,2], [-1,5], [2,4], [9,0.5]]
        }
        return cls(params)

class CorridorMAB(Environment):

    def __init__(self, params):
        '''
        Initializes the Corridor of Multi Armed Bandits environment

        Inputs: params -> dict
                must have attributes:
                    'num_rooms': number of consecutive rooms : int
                    'num_bandits': number of bandits per room : int
                    'reward_distrib': distributions of bandits : list
                    'reward_distrib_params': distributions parameters : list of lists (specifies the best bandit in each room)
        '''

        self.num_rooms   = params['num_rooms']
        self.num_bandits = params['num_bandits']
        
        self.reward_distrib        = params['reward_distrib'] 
        self.reward_distrib_params = params['reward_distrib_params']
        
        self.move_penalty = params['move_penalty']

        if self.num_bandits > len(self.reward_distrib) and len(self.reward_distrib) == 1:
            self.reward_distrib *= self.num_bandits

        assert len(self.reward_distrib) == self.num_bandits, "Too few distributions specified for that number of bandits"

        # Populate that distribution through rooms
        self.reward_distrib = [self.reward_distrib]*self.num_rooms

        # Populate reward_distrib_params
        new_params = []
        for mu, var in self.reward_distrib_params:
            this_state = [[mu, var]]
            for _ in range(self.num_bandits - 1):
                this_state.append([
                    mu - max(0, random.gauss(1,1)) - 1,
                    var + max(-var, random.gauss(0,2)) + 1,
                ])
            new_params.append(this_state) 

        self.reward_distrib_params = new_params
        
        # Mount an environment object
        super(CorridorMAB, self).__init__()
    
    def get_dynamics(self):
        '''
        Returns a function D:(s,a) -> next_state
        '''
        def D(s, a):
            if a == self.num_bandits: # This means go to room to the left
                if s == 0:
                    return s
                else:
                    return s - 1 
            
            if a == (self.num_bandits + 1): # This means go to room to the right
                if s == (self.num_rooms - 1):
                    return s
                else:
                    return s + 1

            if a < self.num_bandits: # This activates one of the arms
                return s
        
        return D

    def get_rewards(self):
        '''
        Returns a function R:(s,a,s_) -> reward

        We make the reward function depend only on s,a
        '''
        def R(s, a, s_):
            if a < self.num_bandits: # This action activates one of the arms
                dist = getattr(Distribution, self.reward_distrib[s][a])
                return dist(*self.reward_distrib_params[s][a])
            else: # This action triggers room change and is associated with a negative reward
                return self.move_penalty

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
        self._states   = np.arange(self.num_rooms)
        self._actions  = np.arange(self.num_bandits + 2)
        self._sa_pairs = [(s,a) for s,a in list(itertools.product(self._states, self._actions))]

    def oracle_agent(self):
        '''
        Returns the cumulative reward of the oracle agent
        at time self.t
        '''
        # Figure out which bandit is the best
        idx = np.unravel_index(np.argmax(self.reward_distrib_params), 
                               np.array(self.reward_distrib_params).shape)

        if self.t <= (idx[0]):
            return self.t * self.move_penalty
        else:
            return idx[0]*self.move_penalty + (self.t - idx[0])*self.reward_distrib_params[idx[0]][0][0]
    
    def get_true_dynamics_matrix(self):
        '''
        Returns the true dynamics matrix
        '''
        D = np.zeros((self.num_rooms,self.num_rooms,self.num_bandits+2))

        dynamics = self.get_dynamics()

        for state in self._states:
            for action in self._actions:
                next_s = dynamics(state, action)
                D[state, next_s, action] = 1

        return D
    
    def get_true_rewards_matrix(self):
        '''
        Returns the true rewards matrix
        '''
        R = np.zeros((self.num_rooms, self.num_bandits + 2))
        R[:, (-1,-2)] = self.move_penalty

        for state in self._states:
            for action in self._actions:
                if action < self.num_bandits:
                    R[state, action] = self.reward_distrib_params[state][action][0]
        
        return R
    
    @classmethod
    def default(cls, **kwargs):
        params = {'num_rooms': 7,
                  'num_bandits': 4, 
                  'reward_distrib': ['normal'], 
                  'reward_distrib_params': [[-1,1], [-2,1], [-3,1], [-4,1], [-5,1], [-6,1], [10,1]],
                  'move_penalty': -20,
        }
        for arg in kwargs.keys():
            params[arg] = kwargs[arg]
        
        return cls(params)

class MazeMAB(Environment):
    '''
    This environment will be a maze with bandits
    '''

