import numpy as np

from sklearn.neighbors import KernelDensity
from copy import deepcopy
from indexedproperty import indexedproperty
import importlib
from scipy import stats
import time

from .agent_utils import Memory
from .models import *
from .action_selection import ActionSelector

class Agent(object):
    
    def __init__(self, gamma, compute_Q_every_step, logger):

        self.gamma                = gamma
        self.compute_Q_every_step = compute_Q_every_step
        
        self.memory_buffer        = []
        self.since_last_Q_compute = self.compute_Q_every_step

        self.logger = logger

        self.dp_maxit = 10000
        
        if logger is not None:
            self.logger_params = logger.memory_params
            self.log_this      = True
    
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

        self.observed_r = {pair:[] for pair in self._sa_pairs}
        self.observed_d = {pair:[] for pair in self._sa_pairs}

        # Initialize random policy
        self.pi = np.ones((len(self._env_states), len(self._env_actions)))
        self.pi = self.pi/self.pi.sum(axis = 1, keepdims=True)

        # If using 'e-random-cheater' policy for comparison of uncert estimates
        # we need to give the agent the optimal Q

        if self.decision_making_method == 'e-random-cheater':
            if not hasattr(env, 'optimal_Q'):
                env.get_true_Q(self.gamma)
            
            self.optimal_Q = env.optimal_Q

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

    def solve_for_Q(self, D, R):
        '''
        Solves for Q given models as dynamics and rewards by policy iteration
        '''

        pi = np.ones((len(self._env_states), len(self._env_actions)))
        pi = pi/pi.sum(axis = 1, keepdims=True)

        converged = False
        it        = 0
        while not converged:
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

            if np.all(pi == pi_) or it >= self.dp_maxit:
                converged = True
            
            pi  = pi_
            it += 1

        return Q, pi
    
    def get_predictive_variance_return(self):
        '''
        Solves for the variance of return (inefficiently)
        '''
        D  = self.get_moment_dynamics_matrix(which_moment = 1)
        R  = self.get_moment_rewards_matrix(which_moment = 1)
        R2 = self.get_moment_rewards_matrix(which_moment = 2)

        Q  = self.get_predictive_Q() 

        ret_sq  = np.zeros((len(self._env_states), len(self._env_actions)))

        converged = False
        while not converged:
            
            ret_sq_ = deepcopy(R2)
            for s_ in self._env_states:
                for a_ in self._env_actions:
                    ret_sq_ += self.pi[s_,a_]*(2*self.gamma*Q[s_,a_]*np.multiply(D[:,s_,:],R) + D[:,s_,:]*(self.gamma**2)*ret_sq[s_,a_])

            if np.max(np.abs(ret_sq - ret_sq_)) < 0.01:
                converged = True
            
            ret_sq = deepcopy(ret_sq_)

        return ret_sq - Q**2
    
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
            Qs[it], _ = self.solve_for_Q(D[it], R[it])

        Q_mean = np.mean(Qs, axis = 0)
        Q_var  = np.var(Qs, axis = 0)

        return Q_mean, Q_var
    
    def get_predictive_Q(self):
        ''' 
        Computes E_{pars}[Q], assuming DAG
        Inputs:
            None
        
        Outputs:
            Q : E_{pars}[Q]
        '''
        # Get predictive matrices
        D = self.get_moment_dynamics_matrix(which_moment = 1)
        R = self.get_moment_rewards_matrix(which_moment = 1)

        Q, pi_ = self.solve_for_Q(D, R)

        self.pi = pi_

        return Q
    
    def get_covariance_Q_old(self):
        ''' 
        Computes Cov(Q(s,a), Q(s',a'))
        '''
        if self.Q is None:
            self.get_Q()
        
        # First we need to get E[Q(s,a)Q(s',a')]
        first_term_ = np.zeros((len(self._env_states), \
                               len(self._env_actions), \
                               len(self._env_states), \
                               len(self._env_actions)))

        R = self.get_moment_rewards_matrix(which_moment = 1)
        D = self.get_moment_dynamics_matrix(which_moment = 1)

        E_Q_E_R = np.einsum('ij,km->ijkm', self.Q, R)

        converged = False
        it = 0
        while not converged:
            first_term = deepcopy(E_Q_E_R)
            second_term = np.zeros((len(self._env_states), len(self._env_actions)))
            for s_ in self._env_states:
                for a_ in self._env_actions:
                    second_term += self.pi[s_,a_]*D[:,s_,:]*first_term_[:,:,s_,a_]
            
            first_term += self.gamma * second_term

            if np.max(np.abs(first_term_ - first_term)) < 0.001 or it >= self.dp_maxit:
                converged = True
            
            first_term_  = first_term
            it += 1
        
        CovQ = first_term - np.einsum('ij,km->ijkm', self.Q, self.Q)

        return CovQ
    
    def get_covariance_Q(self):
        ''' 
        Computes Cov(Q(s,a), Q(s',a'))
        '''
        if self.Q is None:
            self.get_Q()
        
        # First we need to get E[Q(s,a)Q(s',a')]
        first_term_ = np.zeros((len(self._env_states), \
                               len(self._env_actions), \
                               len(self._env_states), \
                               len(self._env_actions)))

        R = self.get_moment_rewards_matrix(which_moment = 1)
        D = self.get_moment_dynamics_matrix(which_moment = 1)

        E_Q_E_R = np.einsum('ij,km->ijkm', self.Q, R)

        converged = False
        it = 0
        while not converged:
            second_term = np.einsum('jd,ijk,abjd->abik', self.pi, D, first_term_)
            
            first_term  = E_Q_E_R + self.gamma * second_term

            if np.max(np.abs(first_term_ - first_term)) < 0.01 or it >= self.dp_maxit:
                converged = True
            
            first_term_  = first_term
            it += 1
        
        CovQ = first_term - np.einsum('ij,km->ijkm', self.Q, self.Q)

        return CovQ

    def get_predictive_u(self):
        '''
        Computes the epistemic uncertainty of return directly
        '''
        if self.Q is None:
            self.get_Q()
        
        # Get all necessary matrices
        var_E_R = self.get_moment_rewards_matrix(which_moment = 'epistemic_variance')
        Q_2     = self.Q**2
        #var_p   = self.get_moment_dynamics_matrix(which_moment = 2)
        #var_p   = var_p - self.get_moment_dynamics_matrix(which_moment = 1)**2
        var_p   = self.get_moment_dynamics_matrix(which_moment = 'epistemic_variance')
        E_p_2   = self.get_moment_dynamics_matrix(which_moment = 1)**2

        converged = False

        #self.pi = np.ones((len(self._env_states), len(self._env_actions)))
        u_ = np.zeros(self.Q.shape)

        it  = 0  
        
        first_term  = np.einsum('ij, ij, aib->ab', self.pi, Q_2, var_p)
        while not converged:
            
            second_term = np.einsum('ij, ij, aib->ab', self.pi, u_, E_p_2)
            third_term  = np.einsum('ij, ij, aib->ab', self.pi, u_, var_p)
            
            u = var_E_R + (self.gamma**2) * (first_term + second_term + third_term)

            if np.max(np.abs(u_ - u)) < 0.01 or it >= self.dp_maxit:
                converged = True
            
            u_  = u
            it += 1 
        
        return u
    
    def get_EUB_u(self):
        '''
        Computes the epistemic uncertainty of return
        using our EUB bound approach
        '''
        if self.Q is None:
            self.get_Q()
        
        # Get all necessary matrices
        var_E_R = self.get_moment_rewards_matrix(which_moment = 'epistemic_variance')
        Q       = self.Q
        var_p   = self.get_moment_dynamics_matrix(which_moment = 'epistemic_variance')
        E_p     = self.get_moment_dynamics_matrix(which_moment = 1)
        Cov_p   = self.get_dynamics_covariance_tensor()
        #Cov_Q   = self.get_covariance_Q() 

        converged = False

        #self.pi = np.ones((len(self._env_states), len(self._env_actions)))
        u_ = np.zeros(self.Q.shape)

        it  = 0

        turner = np.ones((len(self._env_states), len(self._env_actions), \
                          len(self._env_states), len(self._env_actions)))
        for s in self._env_states:
            for a in self._env_actions:
                turner[s,a,s,a] = 0

        while not converged:
            
            first_term  = np.einsum('ij, ij, aib->ab', self.pi, Q**2, var_p)
            second_term = np.einsum('ij, ij, aib->ab', self.pi, u_, E_p**2)
            third_term  = np.einsum('ij, ij, aib->ab', self.pi, u_, var_p)

            #bound_term  = np.einsum('ijkl, ij, ijkl, ikab->ab', turner, self.pi**2, np.abs(Cov_Q), np.abs(Cov_p))
            #bound_term += np.einsum('ijkl, ij, ijkl, aib, akb->ab', turner, self.pi**2, np.abs(Cov_Q), E_p, E_p)
            #bound_term += np.einsum('ijkl, ij, ij, kl, ikab->ab', turner, self.pi**2, Q, Q, np.abs(Cov_p))
            
            #bound_term  = np.einsum('ijkl, ij, ij, kl, ikab->ab', turner, self.pi**2, np.sqrt(var_E_R), np.sqrt(var_E_R), np.abs(Cov_p))
            #bound_term += np.einsum('ijkl, ij, ij, kl, aib, akb->ab', turner, self.pi**2, np.sqrt(var_E_R), np.sqrt(var_E_R), E_p, E_p)
            #bound_term += np.einsum('ijkl, ij, ij, kl, ikab->ab', turner, self.pi**2, Q, Q, np.abs(Cov_p))
            
            u = var_E_R + (self.gamma**2) * (first_term + second_term + third_term) #+ 2*(self.gamma**2)*bound_term
            
            if np.max(np.abs(u_ - u)) < 0.01 or it >= self.dp_maxit:
                converged = True
            
            u_  = u
            it += 1 
        
        bound_term  = np.einsum('ijkl, ij, ij, kl, ikab->ab', turner, self.pi**2, np.sqrt(u), np.sqrt(u), np.abs(Cov_p))
        bound_term += np.einsum('ijkl, ij, ij, kl, aib, akb->ab', turner, self.pi**2, np.sqrt(u), np.sqrt(u), E_p, E_p)
        bound_term += np.einsum('ijkl, ij, ij, kl, ikab->ab', turner, self.pi**2, Q, Q, np.abs(Cov_p))

        #bound_term  = np.einsum('ijkl, ij, ijkl, ikab->ab', turner, self.pi**2, np.abs(Cov_Q), np.abs(Cov_p))
        #bound_term += np.einsum('ijkl, ij, ijkl, aib, akb->ab', turner, self.pi**2, np.abs(Cov_Q), E_p, E_p)
        #bound_term += np.einsum('ijkl, ij, ij, kl, ikab->ab', turner, self.pi**2, Q, Q, np.abs(Cov_p))

        #bound_term  = np.einsum('ijkl, ij, ijkl, ikab->ab', turner, self.pi**2, Cov_Q, Cov_p)
        #bound_term += np.einsum('ijkl, ij, ijkl, aib, akb->ab', turner, self.pi**2, Cov_Q, E_p, E_p)
        #bound_term += np.einsum('ijkl, ij, ij, kl, ikab->ab', turner, self.pi**2, Q, Q, Cov_p)

        u += 2*(self.gamma**2)*bound_term

        #u = np.abs(u)
        
        return u
    
    def get_UBE_u(self):
        '''
        Computes the epistemic uncertainty of return
        using the approach in UBE paper
        '''
        if self.Q is None:
            self.get_Q()

        # Get all necessary matrices
        var_p = self.get_moment_dynamics_matrix(which_moment = 'epistemic_variance')
        E_p   = self.get_moment_dynamics_matrix(which_moment = 1)

        # First we calculate local uncertainty
        local_u = self.get_moment_rewards_matrix(which_moment = 'epistemic_variance')
        
        #local_u += (Q_max**2) * np.einsum('ijk, ijk->ij', var_p, 1/E_p)
        
        Q_max = self.u_method_params * (self.gamma/(1-self.gamma)) # This is the Q_max term that bounds Q, we arbitrarily set it to 10 here
        for s_ in self._env_states:
            local_u += (Q_max**2) * var_p[:,s_,:]/E_p[:,s_,:]
        
        # Now we compute the rest using DP
        converged = False

        it  = 0
        u_ = np.zeros(self.Q.shape)
        while not converged:
            u = deepcopy(local_u)
            second_term = np.einsum('ij,aib,ij->ab', self.pi, E_p, u_)
            
            u += (self.gamma**2) * second_term

            if np.max(np.abs(u_ - u)) < 0.01 or it >= self.dp_maxit:
                converged = True

            u_  = u
            it += 1
        
        return u

    def get_reduction_u(self, state):
        '''
        Computes the naive estimate of the reduction in uncertainty
        '''

        if self.u_method == 'predictive' and self.red_method == 'fast':
            return self.get_predictive_reduction_u(state)
        
        if self.u_method == 'EUB' and self.red_method == 'fast':
            return self.get_EUB_reduction_u(state)
        
        if self.u_method == 'UBE' and self.red_method == 'fast':
            return self.get_UBE_reduction_u(state)

        self.log_this = False
        # Available actions
        actions = np.array([pair[1] for pair in self._sa_pairs if pair[0] == state])

        new_uncert = np.zeros(len(actions))

        old_u = deepcopy(self.u)

        copy_R = deepcopy(self._R_)
        copy_D = deepcopy(self._D_)

        for a in actions:
            old_probability = copy_D[(state, a)].get_predictive_moment(1)
            mean_r          = copy_R[(state, a)].get_predictive_moment(1)
            for s_ in self._env_states:

                self._R_[(state, a)].update(self.observed_r[(state, a)] + [mean_r])

                self._D_[(state, a)].update(self.observed_d[(state, a)] + [s_])

                # Now recompute u
                self.get_u()

                new_uncert[a] += old_probability[s_] * self.u[state, a]

                # Reset the models
                self._R_ = deepcopy(copy_R)
                self._D_ = deepcopy(copy_D)
        
        self.u = old_u
        self.log_this = True
        
        return old_u[state, :] - new_uncert

    def get_EUB_reduction_u(self, state):

        self.log_this = False
        # Available actions
        actions = np.array([pair[1] for pair in self._sa_pairs if pair[0] == state])

        new_uncert = np.zeros(len(actions))

        old_u = deepcopy(self.u)

        copy_R = deepcopy(self._R_)
        copy_D = deepcopy(self._D_)

        for a in actions:

            self._R_[(state, a)].update(self.observed_r[(state, a)] + [copy_R[(state, a)].get_predictive_moment(1)])

            self._D_[(state, a)].update(self.observed_d[(state, a)] + [copy_D[(state, a)].get_most_probable_outcome()])

            # Now recompute u
            self.get_u()

            new_uncert[a] += self.u[state, a]

            # Reset the models
            self._R_ = deepcopy(copy_R)
            self._D_ = deepcopy(copy_D)
        
        self.u = old_u
        self.log_this = True
        
        return old_u[state, :] - new_uncert


    def get_UBE_reduction_u(self, state):
        
        #Computes the reduction in predictive uncertainty from a given state

        #It uses the formula for predictive u
        
        # Available actions
        actions = np.array([pair[1] for pair in self._sa_pairs if pair[0] == state])

        # Firstly populate all necessary vectors
        red_var_1 = np.zeros(len(actions)) # thats the rewards reduction

        old_E     = np.zeros((len(self._env_states), len(actions)))
        old_v     = np.zeros((len(self._env_states), len(actions)))

        new_E     = np.zeros((len(self._env_states), len(actions)))
        new_v     = np.zeros((len(self._env_states), len(actions)))

        Q_max     = self.u_method_params*(self.gamma/(1-self.gamma)) 
        
        copy_R_1 = deepcopy(self._R_)
        copy_D_1 = deepcopy(self._D_)
        for a in actions:
            copy_R = deepcopy(self._R_)

            red_var_1[a] = copy_R[(state, a)].get_predictive_moment('epistemic_variance')
            copy_R[(state, a)].update(self.observed_r[(state, a)] + [copy_R[(state, a)].get_predictive_moment(1)])

            red_var_1[a] -= copy_R[(state, a)].get_predictive_moment('epistemic_variance')

            old_E[:, a] = copy_D_1[(state, a)].get_predictive_moment(1)
            old_v[:, a] = copy_D_1[(state, a)].get_predictive_moment('epistemic_variance')
            

            for s_ in self._env_states:
                copy_D = deepcopy(copy_D_1)

                copy_D[(state, a)].update(self.observed_d[(state, a)] + [s_])

                new_E[:, a] += old_E[s_,a]*copy_D[(state, a)].get_predictive_moment(1)
                new_v[:, a] += old_E[s_,a]*copy_D[(state, a)].get_predictive_moment('epistemic_variance')

        self._R_ = deepcopy(copy_R_1)
        self._D_ = deepcopy(copy_D_1)

        if self.Q is None:
            self.get_Q()
        
        #local_red = 0
        #for s_ in self._env_states:
        #    local_red += (Q_max**2)*(old_v[s_,:]/old_E[s_,:] - new_v[s_,:]/new_E[s_,:])
        #local_red +=  red_var_1

        local_red = red_var_1 + (Q_max**2)*np.einsum('ij->j', old_v/old_E - new_v/new_E)
        C         = old_E[state, :] - new_E[state, :]

        u_red = local_red/(1-(self.gamma**2)*self.pi[state,:]*C)

        return u_red 


    def get_predictive_reduction_u(self, state):
        
        #Computes the reduction in predictive uncertainty from a given state

        #It uses the formula for predictive u
        
        # Available actions
        actions = np.array([pair[1] for pair in self._sa_pairs if pair[0] == state])

        # Firstly populate all necessary vectors
        red_var_1 = np.zeros(len(actions))
        red_var_2 = np.zeros((len(self._env_states), len(actions)))
        red_var_3 = np.zeros(len(actions))
        
        copy_R_1 = deepcopy(self._R_)
        copy_D_1 = deepcopy(self._D_)
        for a in actions:
            copy_R = deepcopy(self._R_)

            red_var_1[a] = copy_R[(state, a)].get_predictive_moment('epistemic_variance')
            copy_R[(state, a)].update(self.observed_r[(state, a)] + [copy_R[(state, a)].get_predictive_moment(1)])

            red_var_1[a] -= copy_R[(state, a)].get_predictive_moment('epistemic_variance')

            red_var_2[:, a] = copy_D_1[(state, a)].get_predictive_moment('epistemic_variance')
            red_var_3[a]    = copy_D_1[(state, a)].get_predictive_moment(1)[state]**2
            
            old_probability = copy_D_1[(state, a)].get_predictive_moment(1)

            for s_ in self._env_states:
                copy_D = deepcopy(copy_D_1)

                copy_D[(state, a)].update(self.observed_d[(state, a)] + [s_])
                
                red_var_2[:, a] -= old_probability[s_] * copy_D[(state, a)].get_predictive_moment('epistemic_variance')
                red_var_3[a]    -= old_probability[s_] * (copy_D[(state, a)].get_predictive_moment(1)[state])**2

        self._R_ = deepcopy(copy_R_1)
        self._D_ = deepcopy(copy_D_1)

        if self.Q is None:
            self.get_Q()

        B = np.zeros(len(actions))
        
        B = np.einsum('ij, ij, ia->a', self.pi, self.Q**2, red_var_2)
        #pi = np.ones((len(self._env_states), len(self._env_actions)))
        #for s in self._env_states:
        #    for a in self._env_actions:
        #        B += self.pi[s,a]*(self.Q[s,a]**2)*red_var_2[s, :]
        
        u_red = red_var_1 + (self.gamma**2)*B
        u_red /= (1 - (self.gamma**2)*self.pi[state,:]*(red_var_2[state, :] + red_var_3))

        return u_red 


    def take_action(self, state):
        '''
        Function returning action in a given state

        Works in two modes, either computes the entire policy and 
        samples from it or just computes action|state
        '''

        if self.since_last_Q_compute >= self.compute_Q_every_step:
            self.get_Q()
            self.get_u()
            self.since_last_Q_compute = 0

        if 'pi' in self.logger_params:

            raise Warning("Uncharted territory")
            
            self.compute_entire_policy()
            a = int(np.where(self.pi[state,:] == 1)[0])
        else:
            a = self.make_decision(state)

        self.since_last_Q_compute += 1

        return a
    
    @property
    def Q(self):
        return self._Q

    @property
    def u(self):
        return self._u
    
    @property
    def pi(self):
        return self._pi

    @u.setter
    def u(self, matrix):
        self.__on_change__('u', matrix)
        self._u = matrix

    @Q.setter
    def Q(self, matrix):
        self.__on_change__('Q', matrix)
        self._Q = matrix
    
    @pi.setter
    def pi(self, matrix):
        self.__on_change__('pi', matrix)
        self._pi = matrix
    
    def __on_change__(self, which, value):
        if self.logger is not None:
            if self.log_this is not False:
                self.logger.update(which, value)
    
    def compute_entire_policy(self):
        '''
        Populates the pi matrix with the most current deterministic policy

        Relies on make_decision routine implemented by Child
        '''
        pi = np.zeros((len(self._env_states), len(self._env_actions)))
        for state in self._env_states:
            action = self.make_decision(state)
            pi[state, action] = 1
        
        self.pi = pi
    
    def compute_greedy_policy(self):
        '''
        Returns the greedy policy of the agent
        '''
        pi = np.zeros((len(self._env_states), len(self._env_actions)))
        for state in self._env_states:
            action = np.argmax(self.Q[state,:])
            pi[state, action] = 1
        
        self.greedy_pi = pi
        
    def get_state_frequencies(self):
        '''
        Returns 10 state visit frequencies 
        '''
        freqs = np.zeros((10, len(self._env_states)))

        total_time = len(self.memory_buffer)
        step       = int(total_time/10)

        for i in range(10):
            f    = np.zeros(len(self._env_states))
            u, c = np.unique(np.array(self.memory_buffer)[:(i+1)*step, 0], return_counts = True)
            f[u.astype(int)] = c
            freqs[i, :] = f/np.sum(f)
        
        return freqs
    
    def get_action_frequencies(self):
        '''
        Returns 10 action frequencies 
        '''
        freqs = np.zeros((10, len(self._env_actions)))

        total_time = len(self.memory_buffer)
        step       = int(total_time/10)

        for i in range(10):
            f    = np.zeros(len(self._env_actions))
            u, c = np.unique(np.array(self.memory_buffer)[:(i+1)*step, 1], return_counts = True)
            f[u.astype(int)] = c
            freqs[i, :] = f/np.sum(f)
        
        return freqs

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
    
    def get_dynamics_covariance_tensor(self, *args, **kwargs):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError

    def get_Q(self, *args, **kwargs):
        '''
        Must be implemented by the child environment
        '''
        raise NotImplementedError

    def get_u(self, *args, **kwargs):
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

class KGAgent(Agent):

    def __init__(self, params):
        '''
        Initializes a simple Knowledge Gradient agent

        This Agent can ONLY play in the MAB setting
        '''

        self.reward_model = params['reward_model']
        self.dyna_model   = params['dyna_model']

        # Initialize name
        self.__name__ = params['name']

        if 'reward_params' in params.keys():
            self.reward_params = params['reward_params']
        
        if 'dyna_params' in params.keys():
            self.dyna_params = params['dyna_params']
    
        super(BayesianAgent, self).__init__(params['gamma'], 0, None)
        
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
                    
                    # This fires the decorator
                    self.R_[pair] = self.R_[pair]
                if self.observed_d[pair]:
                    self.D_[pair].update(self.observed_d[pair])

                    # This fires the decorator
                    self.D_[pair] = self.D_[pair]
    
    def get_Q(self):
        '''
        Populates the agent class with Q estimation
        '''
        pass
    
    def get_u(self):
        '''
        Populates agent class with u estimation (epistemic uncertainty of return)
        '''
        pass

    def make_decision(self, state):
        '''
        Makes a decision in the current state using the KG method
        '''
        pass

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
        self.Q_method         = params['Q_method']
        self.Q_method_params  = params['Q_method_params']

        self.u_method         = params['u_method']
        self.u_method_params  = params['u_method_params']

        # Initialize decision making methods
        self.decision_making_method        = params['decision_making_method']
        self.decision_making_method_params = params['decision_making_method_params']

        # Initialize name
        self.__name__ = params['name']

        self.red_method = params['red_method']

        # Initialize logger 
        if params['logger'] == True:
            self.logger = Memory(params['logger_params'])
        else:
            self.logger = None

        if 'reward_params' in params.keys():
            self.reward_params = params['reward_params']
        
        if 'dyna_params' in params.keys():
            self.dyna_params = params['dyna_params']

        if 'action_list' in params.keys():
            self.action_list = params['action_list']
    
        super(BayesianAgent, self).__init__(params['gamma'], params['compute_Q_every_step'], self.logger)
        
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
                self.D_[pair] = dyna_class({'c': [self.dyna_params]*len(self._env_states),
                                            'out_size': len(self._env_states)})
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

        if hasattr(self, 'logger'):
            #self.logger.update(which, key, value)
            pass
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
                    
                    # This fires the decorator
                    self.R_[pair] = self.R_[pair]
                if self.observed_d[pair]:
                    self.D_[pair].update(self.observed_d[pair])

                    # This fires the decorator
                    self.D_[pair] = self.D_[pair]
    
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
    
    def get_dynamics_covariance_tensor(self):
        '''
        Gets the dynamics covariance tensor

        Return: Covariance tensor of shape (|S|, |S|, |S|, |A|)
        '''
        Cov_tensor = np.zeros((len(self._env_states), len(self._env_states), len(self._env_states), len(self._env_actions)))
        for pair in self._D_.keys():
            for s1 in self._env_states:
                for s2 in self._env_states:
                    Cov_tensor[s1, s2, pair[0], pair[1]] = self._D_[pair].get_state_covariance(s1,s2)
        
        return Cov_tensor
    
    def get_Q(self):
        '''
        Populates the agent class with Q estimation
        '''
        if self.Q_method == 'monte_carlo':
            self.Q, self.u = self.get_monte_carlo_Q(self.Q_method_params)
        
        if self.Q_method == 'predictive':
            self.Q = self.get_predictive_Q()
    
    def get_u(self):
        '''
        Populates agent class with u estimation (epistemic uncertainty of return)
        '''
        if self.u_method == 'predictive':
            self.u = self.get_predictive_u()
        if self.u_method == 'UBE':
            self.u = self.get_UBE_u()
        if self.u_method == 'EUB':
            self.u = self.get_EUB_u()
        if self.u_method == 'monte_carlo':
            _, self.u = self.get_monte_carlo_Q(self.u_method_params)

    def make_decision(self, state):
        '''
        Makes a decision in the current state, given decision making method
        '''

        selector = ActionSelector(self)
        
        return int(selector.select_action(state))
        
    def get_cumulative_reward(self):
        '''
        Returns the cumulative reward obtained so far
        '''
        return np.sum([situation[-1] for situation in self.memory_buffer])

    @classmethod
    def default(cls, **kwargs):
        params = {
            'reward_model': 'BayesianGaussian',
            'dyna_model': 'BayesianCategorical',
            'gamma': 0.95,
            'Q_method': 'monte_carlo',
            'u_method': None,
            'Q_method_params': 20,
            'u_method_params': 10,
            'decision_making_method': 'MCN-KG',
            'decision_making_method_params': 2, 
            'compute_Q_every_step': 1,
            'logger': True,
            'logger_params': ['Q', 'u'],
            'name': None,
            'red_method': 'fast',
        }
        for arg in kwargs.keys():
            params[arg] = kwargs[arg]
            
        return cls(params)
        
    
