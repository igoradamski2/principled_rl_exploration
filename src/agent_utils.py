import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import itertools
import os

#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


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

    def run_experiments(self, steps, num_repeats, fair_comparison = False):
        '''
        Runs num_repeats experiments for steps steps
        for each agent in the environment
        '''
        actions_per_iter = np.zeros((num_repeats, steps))
        if fair_comparison == True:
            # Fair comparison samples a random set of actions
            # for each iteration so that at all runs
            # each agent makes the same actions
            for i in range(num_repeats):
                actions_per_iter[i, :] = np.random.choice(self.env._actions, size = steps)

        for agent_name in self.list_agents():
            
            regrets     = np.zeros((num_repeats, steps))
            best_a      = np.zeros((num_repeats, steps))
            state_freq  = np.zeros((num_repeats, 10, len(self.env._states)))
            step_time   = np.zeros((num_repeats, steps))
            first_t_opt = np.zeros(num_repeats)

            Qs         = np.zeros((num_repeats, steps, len(self.env._states), len(self.env._actions)))
            us         = np.zeros((num_repeats, steps, len(self.env._states), len(self.env._actions)))

            for iteration in range(num_repeats):

                agent = deepcopy(getattr(self, agent_name))
                
                agent.action_list = actions_per_iter[iteration, :]

                if self.env.seed is not None:
                    self.env.seed = iteration*1234

                agent = self.env.play_episode(steps, agent)

                regrets[iteration, :]       = agent.regret
                best_a[iteration, :]        = agent.best_action
                state_freq[iteration, :, :] = agent.get_state_frequencies()
                step_time[iteration, :]     = self.env.elapsed_time_per_step

                where_1 = np.where(agent.is_pi_optimal == 1)[0]
                if len(where_1) == 0:
                    first_t_opt[iteration] = steps
                else:
                    first_t_opt[iteration]      = np.min(where_1)

                Qs[iteration, :, :, :]      = np.array(agent.logger['Q'])
                if 'u' in agent.logger.keys():
                    us[iteration, :, :, :]  = np.array(agent.logger['u'])
            
            agent.mean_regret      = np.mean(regrets, axis = 0)
            agent.sd_regret        = np.sqrt(np.var(regrets, axis = 0))
            
            agent.mean_state_freq  = np.mean(state_freq, axis = 0)
            agent.sd_state_freq    = np.sqrt(np.var(state_freq, axis = 0))

            agent.mean_best_action = np.mean(best_a, axis = 0)
            agent.sd_best_action   = np.sqrt(np.var(best_a, axis = 0))

            agent.mean_step_time   = np.mean(step_time, axis = 0)
            agent.sd_step_time     = np.sqrt(np.var(step_time, axis = 0))

            agent.mean_first_t_opt = np.mean(first_t_opt)
            agent.sd_first_t_opt   = np.var(first_t_opt)

            agent.mean_Qs          = np.mean(Qs, axis = 0)
            agent.sd_Qs            = np.sqrt(np.var(Qs, axis = 0))

            agent.mean_us          = np.mean(us, axis = 0)
            agent.sd_us            = np.sqrt(np.var(us, axis = 0))

            setattr(self, agent_name, agent)
    
    def plot_regret(self, list_agents = None, color_codes = None, figsize = (12,8), title = None, legend_codes = None):
        '''
        Plots regret and % of best action
        '''
        if list_agents is None:
            list_agents = self.list_agents()
        
        fig    = plt.figure(figsize = figsize)
        ax_reg = fig.add_subplot(211)
        ax_ba  = fig.add_subplot(212)

        fig.suptitle(title, fontsize=25)

        for agent_name in list_agents:
            agent = getattr(self, agent_name)
            ax_reg.plot(agent.mean_regret, label = legend_codes[agent_name],
                        color = color_codes[agent_name])
            ax_reg.fill_between(np.arange(len(agent.mean_regret)),
                                agent.mean_regret-agent.sd_regret,
                                agent.mean_regret+agent.sd_regret,
                                alpha=0.33, linestyle = 'dashed',
                                color = color_codes[agent_name])
            
            ax_ba.plot(agent.mean_best_action, label = legend_codes[agent_name],
                       color = color_codes[agent_name])
            ax_ba.fill_between(np.arange(len(agent.mean_best_action)),
                                agent.mean_best_action-agent.sd_best_action,
                                agent.mean_best_action+agent.sd_best_action,
                                alpha=0.33, linestyle = 'dashed',
                                color = color_codes[agent_name])
            
            ax_reg.axvline(x = agent.mean_first_t_opt,
                           color = color_codes[agent_name])
        
        ax_reg.set_title('Online regret', fontsize = 16)
        ax_ba.set_title('% of time best action is chosen', fontsize = 16)

        ax_reg.legend(fontsize = 16)
        ax_ba.legend(fontsize = 16)

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

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

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

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
    
    def plot_Q_u_comparison(self, figsize = (12, 12), color_codes = None, from_t = 0):
        '''
        Plots Q and u but also draws comparison to the monte_carlo agent

        Agents must be named monte_carlo, EUB, EUDV and UCB
        '''
        list_agents = self.list_agents()
        
        assert 'monte_carlo' in list_agents, 'A monte carlo agent needs to be in agent list to draw comparison'

        num_states, num_actions = self.env.optimal_Q.shape

        state_actions = [(i,j) for i in range(num_states) for j in range(num_actions)]

        figures = {}

        monte_carlo_agent = getattr(self, 'monte_carlo')

        u_mc = monte_carlo_agent.mean_us[from_t:]

        if color_codes is None:
            color_codes = {'EUB': '#f30894',
                           'EUDV': '#3138fb',
                           'UBE': '#f23a14'} 

        for agent_name in list_agents:

            if agent_name == 'monte_carlo':
                continue

            fig, axes = plt.subplots(num_states, num_actions, figsize=figsize, \
                                     facecolor='w', edgecolor='k')

            fig.subplots_adjust(hspace = .3, wspace=.2)

            fig.suptitle(agent_name + ' estimate', fontsize=25)

            axes = axes.ravel()

            agent = getattr(self, agent_name)

            #Q = np.array(agent.logger['Q'])
            #u = np.array(agent.logger['u'])

            Q  = agent.mean_Qs[from_t:] 
            u  = agent.mean_us[from_t:]

            for idx, sa in enumerate(state_actions):
                
                axes[idx].plot(Q[:, sa[0], sa[1]], color = '#3138fb', label = r'$\mathbb{E}_{\theta}[Q^{*,\mathcal{W}|\theta_t}]$')
                axes[idx].fill_between(np.arange(Q.shape[0]),
                                       Q[:, sa[0], sa[1]] - 2*np.sqrt(u[:, sa[0], sa[1]]),
                                       Q[:, sa[0], sa[1]] + 2*np.sqrt(u[:, sa[0], sa[1]]),
                                            alpha=0.43, linestyle = 'None', color = color_codes[agent_name],
                                            label = r'Var$_{\theta}[Q^{*,\mathcal{W}|\theta_t}]^{1/2}$')
                
                axes[idx].plot(Q[:, sa[0], sa[1]] - 2*np.sqrt(u_mc[:, sa[0], sa[1]]), 
                              color = 'red', label = 'Monte Carlo uncertainty estimate using $10,000$ samples',
                              linestyle = '--', alpha = 0.8)
                axes[idx].plot(Q[:, sa[0], sa[1]] + 2*np.sqrt(u_mc[:, sa[0], sa[1]]), 
                              color = 'red',
                              linestyle = '--')
                axes[idx].set_title('Room {}, Action {}'.format(sa[0], sa[1]), fontsize = 16)

                bottom = np.min(Q[:, sa[0], sa[1]] - 2.1*np.sqrt(u[:, sa[0], sa[1]]))
                top    = np.max(Q[:, sa[0], sa[1]] + 2.1*np.sqrt(u[:, sa[0], sa[1]]))
                
                bottom = min(bottom, self.env.optimal_Q[sa[0], sa[1]] - 10)
                top    = max(top, self.env.optimal_Q[sa[0], sa[1]] + 10)

                axes[idx].set_ylim(bottom = bottom, 
                                   top    = top)

                axes[idx].set_xticklabels(np.arange(Q.shape[0]) + from_t)

                if idx >= (len(state_actions) - num_actions):
                    axes[idx].set_xlabel('t', fontsize = 16)

                if idx % num_actions == 0:
                    axes[idx].set_ylabel(r'$Q^{*,\mathcal{W}|\theta_t}$', fontsize = 16) 

                # Plot optmial Q for reference
                axes[idx].axhline(y=self.env.optimal_Q[sa[0], sa[1]], 
                                  color = 'black', linestyle = '--', 
                                  label = r'$Q^{*,\mathcal{W}}$', alpha = 0.7)

            handles, labels = axes[idx].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', fontsize = 16, bbox_to_anchor=(0.5, 0))

            plt.tight_layout(rect=[0, 0.22, 1, 0.95])

            figures[agent_name] = fig

            fig.savefig(self.foldername + '/' + agent_name + '_Q')

            plt.show()
        
        return figures

    def plot_Q_u_comparison_one_plot(self, figsize = (12, 12), color_codes = None, from_t = 0, list_agents = None):
        '''
        Plots Q and u but also draws comparison to the monte_carlo agent

        Agents must be named monte_carlo, EUB, EUDV and UCB
        '''
        if list_agents is None:
            list_agents = self.list_agents()

        num_states, num_actions = self.env.optimal_Q.shape

        state_actions = [(i,j) for i in range(num_states) for j in range(num_actions)]

        if color_codes is None:
            color_codes = {'EUB': '#f30894',
                           'EUDV': '#3138fb',
                           'UBE': '#f23a14',
                           'monte_carlo': 'red'}

        agent_names = {'EUB': 'EUB',
                       'EUDV': 'EUDV',
                       'UBE': 'UBE',
                       'monte_carlo': 'Monte Carlo'} 

        fig, axes = plt.subplots(num_states, num_actions, figsize=figsize, \
                                     facecolor='w', edgecolor='k')

        fig.subplots_adjust(hspace = .3, wspace=.2)

        fig.suptitle('Uncertainty estimates comparison', fontsize=25)

        axes = axes.ravel()

        for idx, sa in enumerate(state_actions):
            
            i = 0
            for agent_name in list_agents:

                agent = getattr(self, agent_name)

                Q  = agent.mean_Qs[from_t:] 
                u  = agent.mean_us[from_t:]

                if i == 0:
                    axes[idx].plot(Q[:, sa[0], sa[1]], color = '#3138fb', label = r'$\mathbb{E}_{\theta}[Q^{*,\mathcal{W}|\theta_t}]$',
                                   alpha = 0.4)
                    # Plot optmial Q for reference
                    axes[idx].axhline(y=self.env.optimal_Q[sa[0], sa[1]], 
                                      color = 'black', linestyle = '--', 
                                      label = r'$Q^{*,\mathcal{W}}$', alpha = 0.8)                
                
                axes[idx].plot(Q[:, sa[0], sa[1]] - 2*np.sqrt(u[:, sa[0], sa[1]]), 
                                linestyle = 'dashdot', color = color_codes[agent_name],
                                label = agent_names[agent_name] + ' estimate of ' + r'Var$_{\theta}[Q^{*,\mathcal{W}|\theta_t}]^{1/2}$')                
                
                axes[idx].plot(Q[:, sa[0], sa[1]] + 2*np.sqrt(u[:, sa[0], sa[1]]), 
                                linestyle = 'dashdot', color = color_codes[agent_name])

                axes[idx].fill_between(np.arange(Q.shape[0]),
                                       Q[:, sa[0], sa[1]] - 2*np.sqrt(u[:, sa[0], sa[1]]),
                                       Q[:, sa[0], sa[1]] + 2*np.sqrt(u[:, sa[0], sa[1]]),
                                            alpha=0.43, linestyle = 'None', color = color_codes[agent_name])

                axes[idx].set_title('Room {}, Action {}'.format(sa[0], sa[1]), fontsize = 16)

                #labels = axes[idx].get_xtickslabels()
                #print(labels)
                #axes[idx].set_xticklabels(labels + from_t)

                if idx >= (len(state_actions) - num_actions):
                    axes[idx].set_xlabel('t', fontsize = 16)
                
                if idx % num_actions == 0:
                    axes[idx].set_ylabel(r'$Q^{*,\mathcal{W}|\theta_t}$', fontsize = 16) 
                
                i += 1


        handles, labels = axes[idx].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize = 16, bbox_to_anchor=(0.5, 0))

        plt.tight_layout(rect=[0, 0.22, 1, 0.95])

        fig.savefig(self.foldername + '/' + agent_name + '_Q')

        plt.show()
        
        return fig


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
        




        


    
    
