import numpy as np
from scipy import stats
from copy import deepcopy

class ActionSelector(object):

    def __init__(self, agent):
        self.agent  = agent   
        self.method = agent.decision_making_method
    
    def select_action(self, state):
        func = getattr(self, self.method)
        return func(state)

    def random(self, state):
        '''
        Select action randomly
        '''
        actions = np.array([pair[1] for pair in self.agent._sa_pairs if pair[0] == state])

        np.random.seed(self.agent.decision_making_method_params)
        action = np.random.choice(actions)

        return action
    
    def e_random_cheater(self, state):
        '''
        Select best optimal action with probability
        '''
        actions = np.array([pair[1] for pair in self.agent._sa_pairs if pair[0] == state])

        epsilon = self.agent.decision_making_method_params
        np.random.seed(int(1000*self.agent.decision_making_method_params))

        if np.random.choice([0,1], p = [epsilon, 1-epsilon]) == 1:
            action = np.argmax(self.agent.optimal_Q[state, :])
        else:
            action = np.random.choice(actions)
        
        return action
    
    def e_greedy(self, state):
        '''
        Epsilon greedy policy
        '''
        actions = np.array([pair[1] for pair in self.agent._sa_pairs if pair[0] == state])

        epsilon = self.agent.decision_making_method_params
        if np.random.choice([0,1], p = [epsilon, 1-epsilon]) == 1:
            action = np.argmax(self.agent.Q[state, :])
        else:
            action = np.random.choice(actions)
        
        return action
    
    def SOSS(self, state):
        '''
        Stochastic Optimal Strategy Search
        '''

        actions = np.array([pair[1] for pair in self.agent._sa_pairs if pair[0] == state])
        p_a     = np.ones(len(actions)) 
        for a in actions:
            for a_ in actions:
                
                if a == a_:
                    continue
                else:
                    p_a[a] *= stats.norm.cdf((self.agent.Q[state,a] - self.agent.Q[state,a_])/np.sqrt((self.agent.u[state,a] + self.agent.u[state,a_])))            
        
        p_a = p_a/np.sum(p_a, keepdims=True)

        action = np.random.choice(a=actions, size=1, p=p_a)

        return action

    def ThompsonSampling(self, state):
        '''
        Thompson sampling
        '''
        actions = np.array([pair[1] for pair in self.agent._sa_pairs if pair[0] == state])
        Q_a     = np.zeros(len(actions))
        
        for a in actions:
            Q_a[a] = self.agent.Q[state, a] + stats.norm.rvs(size = 1) * np.sqrt(self.agent.u[state, a])            
        
        action = np.argmax(Q_a)

        return action
    
    def MultivariateTS(self, state):
        '''
        Multivariate Thompson Sampling
        '''
        actions = np.array([pair[1] for pair in self.agent._sa_pairs if pair[0] == state])
            
        CovQ = self.agent.get_covariance_Q()
        # Flatten Q and CovQ
        fl_Q = self.agent.Q.reshape(-1)
        CovQ = CovQ.reshape(len(self.agent._env_states)*len(self.agent._env_actions), \
                            len(self.agent._env_states)*len(self.agent._env_actions))
        
        #CovQ[np.diag_indices_from(CovQ)] = self.u.reshape(-1)            
        # Sample
        
        true_Q = np.random.multivariate_normal(mean = fl_Q, cov = CovQ)
        true_Q = true_Q.reshape(len(self.agent._env_states), len(self.agent._env_actions))
        
        action = np.argmax(true_Q[state, :])

        return action
    
    def UCB(self, state):
        '''
        Upper Confidence Bound
        '''
        action = np.argmax(self.agent.Q[state,:] + self.agent.decision_making_method_params * self.agent.u[state,:])

        return action
    
    def GKG1(self, state):
        '''
        Generalized 1-step Knowledge Gradient
        '''
        actions = np.array([pair[1] for pair in self.agent._sa_pairs if pair[0] == state])

        tilda_u     = self.agent.get_predictive_reduction_u(state)

        influence   = np.zeros(len(actions))
        
        for a in actions:
            influence[a] = -np.abs((self.agent.Q[state, a] - np.max(np.delete(self.agent.Q[state, :], a)))/tilda_u[a])
        
        f_influence = influence*stats.norm.cdf(influence) + stats.norm.pdf(influence)
        kg          = f_influence * tilda_u
        action      = np.argmax(kg) 

        return action
    
    def GKG1_BO(self, state):
        '''
        Generalized Knowledge Gradient using BO formula
        '''
        actions = np.array([pair[1] for pair in self.agent._sa_pairs if pair[0] == state])
        
        tilda_u     = self.agent.get_predictive_reduction_u(state)            
        
        influence = (self.agent.Q[state, :] - np.max(self.agent.Q[state, :]))/tilda_u
        
        f_influence = influence*stats.norm.cdf(influence) + stats.norm.pdf(influence)
        
        kg          = f_influence * tilda_u
        
        action      = np.argmax(kg) 

        return action

    def MCKG1(self, state):
        '''
        Monte Carlo 1-step Knowledge Gradient
        '''
        self.agent.log_this = False
        # First get actions available in this state
        actions = np.array([pair[1] for pair in self.agent._sa_pairs if pair[0] == state])
        kg_return = np.zeros(len(actions))
        
        for a in actions:
            # Step 1: Remember the previous mean reward and models
            kg_return[a] = self.agent.R_[(state, a)].get_predictive_moment(which_moment = 1)
            old_R        = deepcopy(self.agent.R_[(state, a)])
            old_D        = deepcopy(self.agent.D_[(state, a)])
            
            # Step 1.5: Sample new reward and update models
            r      = self.agent.R_[(state, a)].sample_reward(num_samples = 1)
            self.agent.R_[(state, a)].update(self.agent.observed_r[(state, a)] + [r])
            
            # Step 2: Sample new transition and update models
            s      = self.agent.D_[(state, a)].sample_state(num_samples = 1)
            self.agent.D_[(state, a)].update(self.agent.observed_d[(state, a)] + [s])
            
            # Step 3: Update Q
            self.agent.get_Q()
            kg_return[a] += self.agent.gamma * np.max(self.agent.Q[s, :])
            
            # Reset models to the state before simulation
            self.agent.R_[(state, a)] = deepcopy(old_R)
            self.agent.D_[(state, a)] = deepcopy(old_D)            
        
        self.agent.log_this = True
        action = np.argmax(kg_return)

        return action
    
    def MCKGN(self, state):
        '''
        Monte Carlo N-step Knowledge gradient
        '''
        self.agent.log_this = False
            
        actions = np.array([pair[1] for pair in self.agent._sa_pairs if pair[0] == state])
        kg_return = np.zeros(len(actions))            
        old_R = deepcopy(self.agent._R_)
        old_D = deepcopy(self.agent._D_)
        
        for a in actions:
            now_s = state
            now_a = a
            for i in range(self.agent.decision_making_method_params - 1):
                
                kg_return[a] += (self.agent.gamma**i) * self.agent.R_[(now_s, now_a)].get_predictive_moment(which_moment = 1)
                
                # Step 1.5: Sample new reward and update models
                r      = self.agent.R_[(now_s, now_a)].sample_reward(num_samples = 1)
                self.agent.R_[(now_s, now_a)].update(self.agent.observed_r[(now_s, now_a)] + [r])
                
                # Step 2: Sample new transition and update models
                s_      = self.agent.D_[(now_s, now_a)].sample_state(num_samples = 1)
                self.agent.D_[(now_s, now_a)].update(self.agent.observed_d[(now_s, now_a)] + [s_])
                
                # Step 3: Update Q
                self.agent.get_Q()
                new_action = int(np.argmax(kg_return + (self.agent.gamma**(i+1)) * np.max(self.agent.Q[s_, :])))
                now_s = int(s_)
                now_a = int(new_action)                
            
            kg_return[a] += (self.agent.gamma**self.agent.decision_making_method_params)*np.max(self.agent.Q[now_s, :])
            
            # Reset models to the state before simulation
            self.agent._R_ = deepcopy(old_R)
            self.agent._D_ = deepcopy(old_D)
        
        self.agent.log_this = True
        action = np.argmax(kg_return)

        return action

    def PSRL(self, state):
        '''
        Posterior sampling action selection
        '''
        # Sample dynamics
        D = np.squeeze(self.agent.get_sampled_moments_dynamics_matrix(which_moment = 0, num_samples = 1))
        R = np.squeeze(self.agent.get_sampled_moments_rewards_matrix(which_moment = 1, num_samples = 1))

        # Solve for Q
        Q, pi = self.agent.solve_for_Q(D, R)

        # Act greedily
        action = np.argmax(Q[state, :])

        return action