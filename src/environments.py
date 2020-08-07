import numpy as np
from copy import deepcopy
import random
import itertools
from tqdm import tqdm
from sys import stdout
from scipy.stats import bernoulli
from time import time
from scipy import special, stats

from .distributions import Distribution


class Environment(object):

    def __init__(self, seed = None):
        
        # Get dynamics and rewards
        self.D = self.get_dynamics()
        self.R = self.get_rewards()

        # Get initial state
        self.s = self.get_initial_state()

        # Get all state action pairs
        self.get_all_state_action_pairs()

        # Set time to 0
        self.t = 0

        # Set seed?
        # if not None then each timestep returns
        # same rewards and transitions for different agents
        self.seed = seed
    
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
        if self.seed is not None:
            np.random.seed(self.t + self.seed)
        
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

        # Keep track of performance
        self.elapsed_time_per_step = np.zeros(num_steps)
        
        # Initialize agent in environment
        agent.initialize_in_environment(self)

        # Compute optimal behaviour in environment
        self.get_true_Q(agent.gamma)

        # Initialize regret
        regret = np.zeros(num_steps)

        # Initialize % of times selecting best action
        best_action = np.zeros(num_steps)

        # Initialize boolean if greedy policy if equal to optimal
        is_pi_optimal        = np.zeros(num_steps)
        is_pi_almost_optimal = np.zeros(num_steps)

        _oracle = self.oracle_agent()
        i = 0
        for step in tqdm(range(num_steps)):
            
            start_t = time()

            # Take an action
            a = agent.take_action(self.s)

            # Observe new state and reward
            s  = deepcopy(self.s) 
            r  = self.step(a)
            s_ = deepcopy(self.s)

            # Update agent
            agent.update([s, a, s_, r])

            # If this is the mystery world we need to recompute Q, pi after some time
            if hasattr(self, 'switch') and i == 0:
                if self.switch > 0:
                    self.get_true_Q(agent.gamma)
                    i = 1
                    is_pi_optimal        = np.zeros(num_steps)
                    is_pi_almost_optimal = np.zeros(num_steps)


            if self.finish_condition():
                self.reset()

            #stdout.write('\nStep {}/{} - I chose action {} in state {}'.format(step, num_steps, a, s))
            #stdout.flush()

            # Calculate regret
            regret[step] = (self.oracle_agent() - agent.get_cumulative_reward())

            # Calculate % of times selecting best action
            best_action[step] = 1 if a == int(np.where(self.optimal_pi[s, :] == 1)[0]) else 0

            # Calculate greedy policy
            agent.compute_greedy_policy()
            is_pi_optimal[step]        = 1 if np.all(agent.greedy_pi == self.optimal_pi) else 0 
            is_pi_almost_optimal[step] = 1 if np.sum(agent.greedy_pi == self.optimal_pi)/self.optimal_pi.size >= 0.9 else 0


            self.elapsed_time_per_step[step] = time() - start_t
        
        agent.regret        = regret
        agent.best_action   = np.cumsum(best_action)/(np.arange(len(best_action))+1)
        
        agent.is_pi_optimal        = is_pi_optimal
        agent.is_pi_almost_optimal = is_pi_almost_optimal
        
        agent.success        = True if is_pi_optimal[-1] == 1 else False
        agent.almost_success = True if is_pi_almost_optimal[-1] else False

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
    
    def finish_condition(self):
        '''
        Must be implemented by the child environment

        Returns True/False with respect to some episode terminating condition
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
    

    def get_true_variance_return(self, gamma):
        '''
        Populates the env with true variance of return value for a particular gamma,
        for the OPTIMAL pi

        D in shape |S| |S'| |A|
        '''
        if not hasattr(self, 'optimal_Q'):
            self.get_true_Q(gamma)
        
        D  = self.get_true_dynamics_matrix().astype(float)
        R  = self.get_true_rewards_matrix().astype(float)
        R2 = self.get_true_sec_moment_rewards_matrix().astype(float)

        Q  = self.optimal_Q.astype(float)
        pi = self.optimal_pi.astype(float)

        ret_sq = np.zeros((len(self._states), len(self._actions)))

        converged = False
        while not converged:
            #first_term = np.einsum('ijk,j', D, 2 * gamma * np.einsum('ij,ij,ij->i', pi, R, Q))
            #sec_term   = np.einsum('ijk,j', D, (gamma**2) * np.einsum('ij,ij->i', pi, ret_sq))
            
            #ret_sq_ = R2 + first_term + sec_term

            ret_sq_ = deepcopy(R2)

            for s_ in self._states:
                for a_ in self._actions:
                    ret_sq_ += pi[s_,a_]*(2*gamma*Q[s_,a_]*np.multiply(D[:,s_,:],R) + D[:,s_,:]*(gamma**2)*ret_sq[s_,a_])

            if np.max(np.abs(ret_sq - ret_sq_)) < 0.01:
                converged = True
            
            ret_sq = ret_sq_
        
        self.optimal_variance_return = ret_sq - Q**2


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
        super(MultiArmedBandit, self).__init__(params['seed'])
         
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
    
    def get_true_sec_moment_rewards_matrix(self):
        '''
        Returns the true second moment rewards matrix
        '''
        R1 = np.array(self.reward_distrib_params)[:,0]
        R2 = np.array(self.reward_distrib_params)[:,1]

        R  = R2 + R1**2 
        R = R.reshape(len(self._states), len(self._actions))
        return R
        
    
    def finish_condition(self):
        '''
        Returns True/False with respect to some episode terminating condition
        '''
        return False
            
    @classmethod
    def default(cls, **kwargs):
        params = {'num_bandits': 10, 
                  'reward_distrib': ['normal'], 
                  'reward_distrib_params': [[1,1], [2,2], [3,0.5], [4,5], [8,2], [3,1], [10,2], [-1,5], [2,4], [9,0.5]],
                  'seed': None,
        }
        for arg in kwargs.keys():
            params[arg] = kwargs[arg]
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
        super(CorridorMAB, self).__init__(params['seed'])
    
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
        idx = np.unravel_index(2*np.argmax(np.array(self.reward_distrib_params)[:,:,0]), 
                               np.array(self.reward_distrib_params).shape)

        assert idx[1] == 0 and idx[2] == 0, 'Something wrong!'
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
    
    def get_true_sec_moment_rewards_matrix(self):
        '''
        Returns the true second moment rewards matrix
        '''
        R = np.zeros((self.num_rooms, self.num_bandits + 2))
        R[:, (-1,-2)] = self.move_penalty**2

        for state in self._states:
            for action in self._actions:
                if action < self.num_bandits:
                    R[state, action] = self.reward_distrib_params[state][action][1] + self.reward_distrib_params[state][action][0]**2
        
        return R
    
    def finish_condition(self):
        '''
        Returns True/False with respect to some episode terminating condition
        '''
        return False
    
    @classmethod
    def default(cls, **kwargs):
        params = {'num_rooms': 7,
                  'num_bandits': 4, 
                  'reward_distrib': ['normal'], 
                  'reward_distrib_params': [[-1,1], [-2,1], [-3,1], [-4,1], [-5,1], [-6,1], [10,1]],
                  'move_penalty': -20,
                  'seed': None,
        }
        for arg in kwargs.keys():
            params[arg] = kwargs[arg]
        
        return cls(params)

class MysteryCorridor(Environment):
    '''
    Mystery corridor environment
    '''
    def __init__(self, params):
        '''
        Initializes the Corridor of Multi Armed Bandits environment

        Inputs: params -> dict
                must have attributes:
                    'num_rooms': number of consecutive rooms, without the mystery room : int
                    'num_bandits': number of bandits per room : int
                    'reward_distrib': distributions of bandits, without the mystery room : list
                    'reward_distrib_params': distributions parameters : list of lists (specifies the best bandit in each room)
                    'mystery_room': reward_distrib_params but for the mystery room
        '''

        self.num_rooms   = params['num_rooms']
        self.num_bandits = params['num_bandits']
        
        self.reward_distrib        = params['reward_distrib'] 
        self.reward_distrib_params = params['reward_distrib_params']

        self.mystery_room = params['mystery_room']
        
        self.move_penalty = params['move_penalty']

        self.switch_reward = params['switch_reward']

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

        # At first, the door to mystery room is blocked
        self.switch = 0
        
        # Mount an environment object
        super(MysteryCorridor, self).__init__(params['seed'])
    
    def reset(self):
        '''
        Reset the environment state and time to the initial state
        '''
        self.s = self.get_initial_state()
        self.t = 0
        self.switch = 0

    def get_dynamics(self):
        '''
        Returns a function D:(s,a) -> next_state
        '''
        def D(s, a):
            if a == self.num_bandits: # This means go to room to the left
                if s == 0:
                    if self.switch == 0:
                        return s
                    else:
                        return self.num_rooms
                #elif s == self.num_rooms:
                #    return self.num_rooms: # uncomment to makes things easier
                else:
                    if s == self.num_rooms:
                        return s
                    return s - 1 
            
            if a == (self.num_bandits + 1): # This means go to room to the right
                if s == (self.num_rooms - 1):
                    if self.switch == 0:
                        self.switch = 1
                    return s
                
                elif s == self.num_rooms:
                    return 0
                
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
                if s != self.num_rooms:
                    dist = getattr(Distribution, self.reward_distrib[s][a])
                    return dist(*self.reward_distrib_params[s][a])
                else:
                    dist = getattr(Distribution, 'normal')
                    return dist(*self.mystery_room[a])

            else: # This action triggers room change and is associated with a negative reward (mostly)
                if s == self.num_rooms - 1 and a == self.num_bandits + 1: # means we are in room with switch
                    if self.switch == 1:
                        self.switch += 1
                        return self.switch_reward
                    elif self.switch == 2:
                        return 0
                
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
        self._states   = np.arange(self.num_rooms + 1)
        self._actions  = np.arange(self.num_bandits + 2)
        self._sa_pairs = [(s,a) for s,a in list(itertools.product(self._states, self._actions))]

    def oracle_agent(self):
        '''
        Returns the cumulative reward of the oracle agent
        at time self.t
        '''
        # Figure out which bandit is the best
        idx = np.unravel_index(2*np.argmax(np.array(self.reward_distrib_params)[:,:,0]), 
                               np.array(self.reward_distrib_params).shape)

        assert idx[1] == 0 and idx[2] == 0, 'Something wrong!'
        if self.t <= self.num_rooms - 1:
            return self.t * self.move_penalty
        elif self.t == self.num_rooms:
            return (self.t - 1) * self.move_penalty + self.switch_reward
        elif self.t <= 2*self.num_rooms:
            return (self.num_rooms - 1) * self.move_penalty + self.switch_reward + (self.t - self.num_rooms) * self.move_penalty
        else:
            return (self.num_rooms - 1) * self.move_penalty + self.switch_reward + (self.num_rooms) * self.move_penalty + (self.t -2*self.num_rooms)*self.mystery_room[0][0]
    
    def get_true_dynamics_matrix(self):
        '''
        Returns the true dynamics matrix
        '''
        D = np.zeros((self.num_rooms+1,self.num_rooms+1,self.num_bandits+2))

        dynamics = self.get_dynamics()

        prev_switch = deepcopy(self.switch)
        for state in self._states:
            for action in self._actions:
                next_s = dynamics(state, action)
                self.switch = prev_switch
                D[state, next_s, action] = 1

        return D
    
    def get_true_rewards_matrix(self):
        '''
        Returns the true rewards matrix
        '''
        R = np.zeros((self.num_rooms+1, self.num_bandits + 2))

        rewards = self.get_rewards()

        for state in self._states:
            for action in self._actions:
                av_reward = np.zeros(100)
                for i in range(100):
                    av_reward[i] = rewards(state, action, None)
                R[state, action] = av_reward.mean()
        
        return R
    
    def get_true_sec_moment_rewards_matrix(self):
        '''
        Returns the true second moment rewards matrix
        '''
        R = np.zeros((self.num_rooms, self.num_bandits + 2))
        R[:, (-1,-2)] = self.move_penalty**2

        for state in self._states:
            for action in self._actions:
                if action < self.num_bandits:
                    R[state, action] = self.reward_distrib_params[state][action][1] + self.reward_distrib_params[state][action][0]**2
        
        return R
    
    def finish_condition(self):
        '''
        Returns True/False with respect to some episode terminating condition
        '''
        return False
    
    @classmethod
    def default(cls, **kwargs):
        params = {'num_rooms': 4,
                  'num_bandits': 2, 
                  'reward_distrib': ['normal'], 
                  'reward_distrib_params': [[0,1], [0,1], [0,1], [1,1]],
                  'mystery_room': [[10,1], [5,1]],
                  'switch_reward': 10,
                  'move_penalty': -1,
                  'seed': None,
        }
        for arg in kwargs.keys():
            params[arg] = kwargs[arg]
        
        return cls(params)

class BinaryTree(Environment):
    '''
    This is the environment from Osband
    '''
    def __init__(self, params):
        '''
        Initiates the environment

        Inputs: params -> dict
            must have attributes:
                    'N': we form a NxN grid of states : int
                    'cost': the cost the agent incurs by moving right : +float (POSITIVE)
                    'final_reward': the reward the agent gets by moving right in rightmost state : list
        '''

        self.N            = params['N'] # Number of layers
        self.cost         = params['cost'] # Cost move right (later divided by N)
        self.final_reward = params['final_reward'] # Reward at the end

        # We need to set a mask for each of the actions
        #W = bernoulli.rvs(0.5, size=2**self.N - 1)

        #uncomment
        W = np.zeros(2**self.N - 1)

        self.W = W

        # Mount an environment object
        super(BinaryTree, self).__init__(params['seed'])
    
    def get_dynamics(self):
        '''
        Returns a function D:(s,a) -> next_state
        '''
        def D(s, a):

            if self.W[s] == 0:
                if a == 0:
                    new_a = 0
                if a == 1:
                    new_a = 1
            
            if self.W[s] == 1:
                if a == 0:
                    new_a = 1
                if a == 1:
                    new_a = 0

            a = new_a
            if a == 0: # Go left
                # If we are at the last level then go to 0
                if s >= (2**(self.N-1) - 1):
                    return 0
                else:
                    return (s+1)*2 - 1

            else:      # Go right
                # If we are at the last level then go to 0
                if s >= (2**(self.N-1) - 1):
                    return 0
                else:
                    return (s+1)*2 

        return D

    def get_rewards(self):
        '''
        Returns a function R:(s,a,s_) -> reward

        We make the reward function depend only on s,a
        '''
        def R(s, a, s_):

            if self.W[s] == 0:
                if a == 0:
                    new_a = 0
                if a == 1:
                    new_a = 1
            
            if self.W[s] == 1:
                if a == 0:
                    new_a = 1
                if a == 1:
                    new_a = 0

            a = new_a
            if a == 0: # Go left

                return 0

            else:      # Go right

                if s == (2**self.N - 2):
                    return self.final_reward
                else:
                    return -self.cost/(self.N-1)

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
        self._states   = np.arange(2**self.N - 1)
        self._actions  = np.arange(2)
        self._sa_pairs = [(s,a) for s,a in list(itertools.product(self._states, self._actions))]

    def oracle_agent(self):
        '''
        Returns the cumulative reward of the oracle agent
        at time self.t
        '''
        comp_eps = np.floor(self.t/self.N)
        return comp_eps * (self.final_reward - (self.N-1)*(self.cost)/(self.N-1)) - ((self.t - (comp_eps*self.N)) * (self.cost/(self.N-1)))
        
    def get_true_dynamics_matrix(self):
        '''
        Returns the true dynamics matrix
        '''
        D = np.zeros((2**self.N - 1, 2**self.N - 1, 2))

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
        R = np.zeros((2**self.N - 1, 2))

        rewards = self.get_rewards()

        for state in self._states:
            for action in self._actions:
                R[state, action] = rewards(state, action, 0)
        
        return R
    
    def finish_condition(self):
        '''
        Returns True/False with respect to some episode terminating condition
        '''
        if self.t == self.N:
            return False
        else:
            return False
    
    @classmethod
    def default(cls, **kwargs):
        params = {'N': 5,
                  'cost': 0.01,
                  'final_reward': 1,
                  'seed': None,
        }
        for arg in kwargs.keys():
            params[arg] = kwargs[arg]
        
        return cls(params)

class ChainMDP(Environment):
    '''
    This is the environment from Osband
    '''
    def __init__(self, params):
        '''
        Initiates the environment

        Inputs: params -> dict
            must have attributes:
                    'N': we form a NxN grid of states : int
                    'cost': the cost the agent incurs by moving right : +float (POSITIVE)
                    'final_reward': the reward the agent gets by moving right in rightmost state : list
        '''

        self.N            = params['N']
        self.cost         = params['cost']
        self.final_reward = params['final_reward']

        # We need to set a mask for each of the actions
        W = bernoulli.rvs(0.5, size=self.N**2)

        #uncomment
        W = np.zeros(self.N**2)

        self.W = W

        # Mount an environment object
        super(ChainMDP, self).__init__(params['seed'])
    
    def get_dynamics(self):
        '''
        Returns a function D:(s,a) -> next_state
        '''
        def D(s, a):

            if self.W[s] == 0:
                if a == 0:
                    new_a = 0
                if a == 1:
                    new_a = 1
            
            if self.W[s] == 1:
                if a == 0:
                    new_a = 1
                if a == 1:
                    new_a = 0

            a = new_a
            if a == 0: # Go left

                if s % self.N == 0: # This means we are at leftmost edge
                    if int(np.floor(s/self.N)) == (self.N - 1): # This means we are at the bottom row
                        return self.get_initial_state()
                    else:
                        return s + self.N # We move down at each step
                else:
                    if int(np.floor(s/self.N)) == (self.N - 1): # This means we are at the bottom row
                        return self.get_initial_state()
                    else:
                        return s - 1 + self.N # We move down at each step

            else:      # Go right

                if s % self.N == (self.N - 1): # This means we are at rightmost edge
                    if int(np.floor(s/self.N)) == (self.N - 1): # This means we are at the bottom row
                        return self.get_initial_state()
                    else:
                        return s + self.N
                else:
                    if int(np.floor(s/self.N)) == (self.N - 1): # This means we are at the bottom row
                        return self.get_initial_state()
                    else:
                        return s + 1 + self.N

        return D

    def get_rewards(self):
        '''
        Returns a function R:(s,a,s_) -> reward

        We make the reward function depend only on s,a
        '''
        def R(s, a, s_):

            if self.W[s] == 0:
                if a == 0:
                    new_a = 0
                if a == 1:
                    new_a = 1
            
            if self.W[s] == 1:
                if a == 0:
                    new_a = 1
                if a == 1:
                    new_a = 0

            a = new_a
            if a == 0: # Go left

                return 0

            else:      # Go right

                if s % self.N == (self.N - 1): # This means we are at rightmost edge
                    return self.final_reward
                else:
                    return -self.cost/self.N

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
        self._states   = np.arange(self.N**2)
        self._actions  = np.arange(2)
        self._sa_pairs = [(s,a) for s,a in list(itertools.product(self._states, self._actions))]

    def oracle_agent(self):
        '''
        Returns the cumulative reward of the oracle agent
        at time self.t
        '''
        comp_eps = np.floor(self.t/self.N)
        return comp_eps * (self.final_reward - (self.N-1)*(self.cost)/self.N) - ((self.t - (comp_eps*self.N)) * (self.cost/self.N))
        
    
    def get_true_dynamics_matrix(self):
        '''
        Returns the true dynamics matrix
        '''
        D = np.zeros((self.N**2, self.N**2, 2))

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
        R = np.zeros((self.N**2, 2))

        rewards = self.get_rewards()

        for state in self._states:
            for action in self._actions:
                R[state, action] = rewards(state, action, 0)
        
        return R
    
    def finish_condition(self):
        '''
        Returns True/False with respect to some episode terminating condition
        '''
        if self.t == self.N:
            return False
        else:
            return False
    
    @classmethod
    def default(cls, **kwargs):
        params = {'N': 5,
                  'cost': 0.01,
                  'final_reward': 1,
                  'seed': None,
        }
        for arg in kwargs.keys():
            params[arg] = kwargs[arg]
        
        return cls(params)

class RandomMDP(Environment):

    def __init__(self, params):
        '''
        Initializes the Random MDP environment

        Inputs: params -> dict
                must have attributes:
                    'mdp_seed': random seed : int (dont confuse with seed)
                    'num_states': int
                    'num_actions' : int
        '''

        self.mdp_seed    = params['mdp_seed']

        # Transition params
        self.c = params['c']
        
        # Rewards params
        self.mu    = params['mu']
        self.lmbda = params['lmbda']
        self.alpha = params['alpha']
        self.beta  = params['beta']

        self.num_states  = params['num_states']
        self.num_actions = params['num_actions']

        self.get_all_state_action_pairs()

        # Sample the environment dynamics
        self._transitions = {}
        self._rewards     = {}
        for idx, pair in enumerate(self._sa_pairs):
            np.random.seed(self.mdp_seed + idx*269)
            
            self._transitions[pair] = stats.dirichlet.rvs(alpha = [self.c]*self.num_states, size = 1)[0]
            
            tau = stats.gamma.rvs(a = self.alpha, scale = 1/self.beta, size = 1)
            m = self.mu + stats.norm.rvs(size = 1)*np.sqrt(1/(self.lmbda*tau))

            self._rewards[pair]     = [m, np.sqrt((1/tau))]
        
        # Mount an environment object
        super(RandomMDP, self).__init__(params['seed'])
    
    def get_dynamics(self):
        '''
        Returns a function D:(s,a) -> next_state
        '''
        def D(s, a):
            np.random.seed()
            return np.random.choice(np.arange(self.num_states), p = self._transitions[(s,a)])
        
        return D

    def get_rewards(self):
        '''
        Returns a function R:(s,a,s_) -> reward

        We make the reward function depend only on s,a
        '''
        def R(s, a, s_):
            np.random.seed()
            return np.random.normal(loc   = self._rewards[(s,a)][0], scale = self._rewards[(s,a)][1])

        return R
    
    def get_initial_state(self):
        np.random.seed(self.mdp_seed)
        return np.random.choice(np.arange(self.num_states))
    
    def get_all_state_action_pairs(self):
        '''
        Returns all possible state-action pairs
            - populate self._states with all state numbers (np.array)
            - populate self._actions with all action numbers (np.array)
            - populate self._sa_pairs with all possible state-action pairs (list of tuples)
        '''
        self._states   = np.arange(self.num_states)
        self._actions  = np.arange(self.num_actions)
        self._sa_pairs = [(s,a) for s,a in list(itertools.product(self._states, self._actions))]

    def oracle_agent(self):
        '''
        Returns the cumulative reward of the oracle agent
        at time self.t assumes that we have optimal_pi
        '''

        if self.t == 0:
            # No steps so no cumulative reward initialized
            optimal_action                 = np.where(self.optimal_pi[self.s, :] == 1)[0][0]
            self._oracle_cumulative_reward = deepcopy(self._rewards[(self.s, optimal_action)][0])
        else:
            optimal_action                  = np.where(self.optimal_pi[self.s, :] == 1)[0][0]
            self._oracle_cumulative_reward += self._rewards[(self.s, optimal_action)][0]
        
        return self._oracle_cumulative_reward
    
    def get_true_dynamics_matrix(self):
        '''
        Returns the true dynamics matrix
        '''
        D = np.zeros((self.num_states,self.num_states,self.num_actions))

        for state in self._states:
            for action in self._actions:
                D[state, :, action] = self._transitions[(state, action)]

        return D
    
    def get_true_rewards_matrix(self):
        '''
        Returns the true rewards matrix
        '''
        R = np.zeros((self.num_states, self.num_actions))

        for state in self._states:
            for action in self._actions:
                R[state, action] = self._rewards[(state, action)][0]
        
        return R
    
    def finish_condition(self):
        '''
        Returns True/False with respect to some episode terminating condition
        '''
        return False
    
    @classmethod
    def default(cls, **kwargs):
        params = {'num_states': 5,
                  'num_actions': 3,
                  'c': 1,
                  'mu': 0,
                  'lmbda': 10,
                  'alpha': 2,
                  'beta': 2,
                  'mdp_seed': 2308,
                  'seed': None,
        }
        for arg in kwargs.keys():
            params[arg] = kwargs[arg]
        
        return cls(params)
    



