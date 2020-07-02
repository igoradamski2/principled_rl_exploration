import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import itertools
import os


class Memory(object):
    '''
    This class serves as a object holder for the memory of the agent
    Specifically this one is created to hold a dictionary of lists
    specifying the distributions at each step per each state-action

    Structure of memory:
        dictionary, with keys corresponding to evolution of particular quantities
    '''

    def __init__(self, memory_params):
        self.memory        = {}
        self.memory_params = memory_params
    
    def __getitem__(self, which):
        return self.memory[which]
    
    def keys(self):
        return self.memory.keys()
    
    def update(self, which, value):
        if which in self.memory_params:
            if which not in self.memory.keys():
                self.memory[which] = [value]
            else:
                self.memory[which].append(value)

class SimplePlotter(object):
    '''
    A plotting class for list of agents

    Mainly used to plot statistically significant regrets

    Has to be called from a place that has a exps/ folder
    '''
    def __init__(self, foldername, env, *agents):
        '''
        env: Environment class
        agents: list of Agent classes
        '''

        assert os.path.isdir('exps') is True, "To call SimplePlotter make sure a exps/ folder exists in same path"

        self.foldername = 'exps/' + foldername
        self.env        = env
        
        for a in agents:
            setattr(self, a.__name__, a)
        
        # If foldername exists and is empty throw error
        if os.path.isdir(self.foldername):
            assert len(os.listdir(self.foldername)) == 0, "Folder given exists and is not empty"

        if not os.path.isdir(self.foldername):
            os.mkdir(self.foldername)
    
    def list_agents(self):
        '''
        Creates a list of agents to iterate through
        '''
        agents = []
        for var in vars(self).keys():
            if var not in ['env', 'foldername', 'alpha']:
                agents.append(var)
        
        return agents

    def run_experiments(self, steps, num_repeats):
        '''
        Runs num_repeats experiments for steps steps
        for each agent in the environment
        '''
        for agent_name in self.list_agents():
            
            regrets    = np.zeros((num_repeats, steps))
            best_a     = np.zeros((num_repeats, steps))
            state_freq = np.zeros((num_repeats, 10, len(self.env._states)))

            Qs         = np.zeros((num_repeats, steps, len(self.env._states), len(self.env._actions)))
            us         = np.zeros((num_repeats, steps, len(self.env._states), len(self.env._actions)))

            for iteration in range(num_repeats):
                agent = deepcopy(getattr(self, agent_name))

                agent = self.env.play_episode(steps, agent)

                regrets[iteration, :]       = agent.regret
                best_a[iteration, :]        = agent.best_action
                state_freq[iteration, :, :] = agent.get_state_frequencies()

                Qs[iteration, :, :, :]      = np.array(agent.logger['Q'])
                if 'u' in agent.logger.keys():
                    us[iteration, :, :, :]  = np.array(agent.logger['u'])
            
            agent.mean_regret      = np.mean(regrets, axis = 0)
            agent.sd_regret        = np.sqrt(np.var(regrets, axis = 0))
            
            agent.mean_state_freq  = np.mean(state_freq, axis = 0)
            agent.sd_state_freq    = np.sqrt(np.var(state_freq, axis = 0))

            agent.mean_best_action = np.mean(best_a, axis = 0)
            agent.sd_best_action   = np.sqrt(np.var(best_a, axis = 0))

            agent.mean_Qs          = np.mean(Qs, axis = 0)
            agent.sd_Qs            = np.sqrt(np.var(Qs, axis = 0))

            agent.mean_us          = np.mean(us, axis = 0)
            agent.sd_us            = np.sqrt(np.var(us, axis = 0))

            setattr(self, agent_name, agent)
    
    def plot_regret(self, *names):
        '''
        Plots regret and % of best action
        '''
        if len(names) == 0:
            list_agents = self.list_agents()
        else:
            list_agents = names
        
        fig    = plt.figure(figsize = (12,8))
        ax_reg = fig.add_subplot(211)
        ax_ba  = fig.add_subplot(212)

        for agent_name in list_agents:
            agent = getattr(self, agent_name)
            ax_reg.plot(agent.mean_regret, label = agent_name)
            ax_reg.fill_between(np.arange(len(agent.mean_regret)),
                                agent.mean_regret-agent.sd_regret,
                                agent.mean_regret+agent.sd_regret,
                                alpha=0.33, linestyle = 'dashed')
            
            ax_ba.plot(agent.mean_best_action, label = agent_name)
            ax_ba.fill_between(np.arange(len(agent.mean_best_action)),
                                agent.mean_best_action-agent.sd_best_action,
                                agent.mean_best_action+agent.sd_best_action,
                                alpha=0.33, linestyle = 'dashed')
        
        ax_reg.set_title('Online regret')
        ax_ba.set_title('% of time best action is chosen')

        ax_reg.legend()
        ax_ba.legend()

        plt.tight_layout()

        fig.savefig(self.foldername + '/regret')
        
        plt.show()
    
    def plot_regret_noerr(self, *names):
        '''
        Plots regret and % of best action
        '''
        if len(names) == 0:
            list_agents = self.list_agents()
        else:
            list_agents = names
        
        fig    = plt.figure(figsize = (12,8))
        ax_reg = fig.add_subplot(211)
        ax_ba  = fig.add_subplot(212)

        for agent_name in list_agents:
            agent = getattr(self, agent_name)
            ax_reg.plot(agent.mean_regret, label = agent_name)
            
            ax_ba.plot(agent.mean_best_action, label = agent_name)
        
        ax_reg.set_title('Online regret')
        ax_ba.set_title('% of time best action is chosen')

        ax_reg.legend()
        ax_ba.legend()

        plt.tight_layout()

        fig.savefig(self.foldername + '/regret_noerr')
        
        plt.show()

    
    def plot_Q_u(self, state_actions = None, *names):
        '''
        Plots the evolution of Q and u on plot grids
        for specified state-action pairs or for all if unspecified
        '''
        if len(names) == 0:
            list_agents = self.list_agents()
        else:
            list_agents = names
        
        num_states, num_actions = self.env.optimal_Q.shape

        if state_actions is None:
            state_actions = [(i,j) for i in range(num_states) for j in range(num_actions)]

        figures = {}

        for agent_name in list_agents:

            fig, axes = plt.subplots(num_states, num_actions, figsize=(20, 20), \
                                     facecolor='w', edgecolor='k')

            fig.subplots_adjust(hspace = .3, wspace=.2)

            plt.tight_layout()

            fig.suptitle(agent_name, fontsize=25)

            axes = axes.ravel()

            agent = getattr(self, agent_name)

            #Q = np.array(agent.logger['Q'])
            #u = np.array(agent.logger['u'])

            Q  = agent.mean_Qs 
            u  = agent.mean_us

            for idx, sa in enumerate(state_actions):
                
                axes[idx].plot(Q[:, sa[0], sa[1]], color = 'blue', label = 'Predicted Q')
                axes[idx].fill_between(np.arange(Q.shape[0]),
                                       Q[:, sa[0], sa[1]] - np.sqrt(u[:, sa[0], sa[1]]),
                                       Q[:, sa[0], sa[1]] + np.sqrt(u[:, sa[0], sa[1]]),
                                            alpha=0.33, linestyle = 'None', color = 'blue')

                axes[idx].set_title('$Q({}, {})$'.format(sa[0], sa[1]))
                #axes[idx].x_label('Iteration') 

                # Plot optmial Q for reference
                axes[idx].axhline(y=self.env.optimal_Q[sa[0], sa[1]], color = 'red', label = 'Optimal Q')

            figures[agent_name] = fig

            fig.savefig(self.foldername + '/' + agent_name + '_Q')

            plt.show()
    
    def plot_state_freq(self, intervals = None, *names):
        '''
        Implement so that it shows in intervals instead of 10
        '''
        if len(names) == 0:
            list_agents = self.list_agents()
        else:
            list_agents = names

        for agent_name in list_agents:

            fig, axes = plt.subplots(2, 5, figsize=(12, 8), \
                                            facecolor='w', edgecolor='k')
            
            axes = axes.ravel()

            agent = getattr(self, agent_name)

            for i in range(10):
                axes[i].bar(agent._env_states, agent.mean_state_freq[i, :])

                axes[i].set_xticks(agent._env_states)

                axes[i].set_title('After step = {}'.format((i+1)*int(len(agent.memory_buffer)/10)))
            
            fig.savefig(self.foldername + '/' + agent_name + '_state_freqs')

            plt.show()
        




        


    
    
