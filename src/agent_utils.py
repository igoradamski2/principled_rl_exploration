import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import itertools
import os

import pickle

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
            action_freq = np.zeros((num_repeats, 10, len(self.env._actions)))
            step_time   = np.zeros((num_repeats, steps))

            Qs          = np.zeros((num_repeats, steps, len(self.env._states), len(self.env._actions)))
            us          = np.zeros((num_repeats, steps, len(self.env._states), len(self.env._actions)))

            success            = []
            almost_success     = []
            first_t_opt        = np.zeros(num_repeats)
            first_t_almost_opt = np.zeros(num_repeats)

            for iteration in range(num_repeats):

                agent = deepcopy(getattr(self, agent_name))
                
                agent.action_list = actions_per_iter[iteration, :]

                if self.env.seed is not None:
                    self.env.seed = iteration*1234

                agent = self.env.play_episode(steps, agent)

                regrets[iteration, :]       = agent.regret
                best_a[iteration, :]        = agent.best_action
                state_freq[iteration, :, :] = agent.get_state_frequencies()
                action_freq[iteration,:,:]  = agent.get_action_frequencies()
                step_time[iteration, :]     = self.env.elapsed_time_per_step

                if agent.success:
                    first_t_opt[iteration] = np.min(np.where(agent.is_pi_optimal == 1)[0])
                    success.append(iteration)
                else:
                    first_t_opt[iteration] = steps
                
                if agent.almost_success:
                    first_t_almost_opt[iteration] = np.min(np.where(agent.is_pi_almost_optimal == 1)[0])
                    almost_success.append(iteration)
                else:
                    first_t_almost_opt[iteration] = steps

                Qs[iteration, :, :, :]      = np.array(agent.logger['Q'])
                if 'u' in agent.logger.keys():
                    us[iteration, :, :, :]  = np.array(agent.logger['u'])
            
            agent.mean_regret      = np.mean(regrets, axis = 0)
            agent.sd_regret        = np.sqrt(np.var(regrets, axis = 0))

            success        = np.array(success)
            almost_success = np.array(almost_success)
            if len(success) > 0:
                agent.mean_regret_s      = np.mean(regrets[success], axis = 0)
                agent.sd_regret_s        = np.sqrt(np.var(regrets[success], axis = 0))

                agent.mean_best_action_s = np.mean(best_a[success], axis = 0)
                agent.sd_best_action_s   = np.var(best_a[success], axis = 0)

                agent.mean_state_freq_s  = np.mean(state_freq[success], axis = 0)
                agent.sd_state_freq_s    = np.var(state_freq[success], axis = 0)

                agent.mean_Qs_s          = np.mean(Qs[success], axis = 0)
                agent.sd_Qs_s            = np.sqrt(np.var(Qs[success], axis = 0))

                agent.mean_us_s          = np.mean(us[success], axis = 0)
                agent.sd_us_s            = np.sqrt(np.var(us[success], axis = 0))

            
            if len(almost_success) > 0:
                agent.mean_regret_a_s      = np.mean(regrets[almost_success], axis = 0)
                agent.sd_regret_a_s        = np.sqrt(np.var(regrets[almost_success], axis = 0))

                agent.mean_best_action_s_a = np.mean(best_a[almost_success], axis = 0)
                agent.sd_best_action_s_a   = np.var(best_a[almost_success], axis = 0)

                agent.mean_state_freq_a_s  = np.mean(state_freq[almost_success], axis = 0)
                agent.sd_state_freq_a_s    = np.var(state_freq[almost_success], axis = 0)

                agent.mean_Qs_a_s          = np.mean(Qs[almost_success], axis = 0)
                agent.sd_Qs_a_s            = np.sqrt(np.var(Qs[almost_success], axis = 0))

                agent.mean_us_a_s          = np.mean(us[almost_success], axis = 0)
                agent.sd_us_a_s            = np.sqrt(np.var(us[almost_success], axis = 0))
            
            agent.mean_state_freq  = np.mean(state_freq, axis = 0)
            agent.sd_state_freq    = np.var(state_freq, axis = 0)

            agent.mean_action_freq  = np.mean(action_freq, axis = 0)
            agent.sd_action_freq    = np.var(action_freq, axis = 0)

            agent.mean_best_action = np.mean(best_a, axis = 0)
            agent.sd_best_action   = np.var(best_a, axis = 0)

            agent.mean_step_time   = np.mean(step_time, axis = 0)
            agent.sd_step_time     = np.sqrt(np.var(step_time, axis = 0))

            agent.mean_Qs          = np.mean(Qs, axis = 0)
            agent.sd_Qs            = np.sqrt(np.var(Qs, axis = 0))

            agent.mean_us          = np.mean(us, axis = 0)
            agent.sd_us            = np.sqrt(np.var(us, axis = 0))

            agent.mean_first_t_opt = np.mean(first_t_opt)
            agent.sd_first_t_opt   = np.sqrt(np.var(first_t_opt))

            agent.mean_first_t_almost_opt = np.mean(first_t_almost_opt, axis = 0)
            agent.sd_first_t_almost_opt   = np.sqrt(np.var(first_t_almost_opt, axis = 0))

            agent.success          = success
            agent.almost_success   = almost_success

            agent.num_repeats      = num_repeats

            setattr(self, agent_name, agent)
    
    def print_performance_metrics(self, list_agents = None):
        ''' 
        Prints performance metrics
        '''

        if list_agents is None:
            list_agents = self.list_agents()
        
        for agent_name in list_agents:
            agent = getattr(self, agent_name)
            
            string  = "Final regret of agent {} is {}".format(agent_name, agent.mean_regret[-1])
            string += "\n Final % of best action of agent {} is {}".format(agent_name, agent.mean_best_action[-1])
            string += "\n Mean successful time is {}".format(agent.mean_first_t_opt)
            string += "\n"

            print("Final regret of agent {} is {}".format(agent_name, agent.mean_regret[-1]))
            print("Final % of best action of agent {} is {}".format(agent_name, agent.mean_best_action[-1]))
            print("Mean successful time is {}".format(agent.mean_first_t_opt))
            print('\n')

            with open(self.foldername + '/report.txt', 'a+') as f:
                f.write(string)
    
    def plot_regret(self, list_agents = None, color_codes = None, 
                    figsize = (12,8), title = None, legend_codes = None,
                    which = 'all', rect = [0, 0.1, 1, 0.9], 
                    bbox = (0.5, -0.25),
                    suptitle_fs = 25, xy_label_fs = 18, 
                    titles_fs = 17.5, ticks_fs = 12.5, legend_fs = 19, 
                    ncol = None, dpi = 200):
        '''
        Plots regret and % of best action
        '''
        if list_agents is None:
            list_agents = self.list_agents()
        
        fig    = plt.figure(figsize = figsize)
        ax_reg = fig.add_subplot(211)
        ax_ba  = fig.add_subplot(212)

        fig.suptitle(title, fontsize=suptitle_fs)

        for agent_name in list_agents:
            agent = getattr(self, agent_name)
            
            if which == 'all':
                regret         = agent.mean_regret
                sd_regret      = agent.sd_regret 

                best_action    = agent.mean_best_action
                sd_best_action = agent.sd_best_action

                num = len(agent.success)
            if which == 'success':
                if hasattr(agent, 'mean_regret_s'):
                    regret    = agent.mean_regret_s
                    sd_regret = agent.sd_regret_s

                    best_action    = agent.mean_best_action_s
                    sd_best_action = agent.sd_best_action_s

                    num = len(agent.success)
                else:
                    continue
            if which == 'almost_success':
                if hasattr(agent, 'mean_regret_s_a'):
                    regret    = agent.mean_regret_a_s
                    sd_regret = agent.sd_regret_a_s

                    best_action    = agent.mean_best_action_a_s
                    sd_best_action = agent.sd_best_action_a_s

                    num = len(agent.almost_success)
                else:
                    continue
            
            add_string = ' ({}/{} successful)'.format(num, agent.num_repeats)

            this_color = np.random.rand(3,)

            ax_reg.plot(regret,
                        color = color_codes[agent_name] if color_codes is not None else this_color)
            ax_reg.fill_between(np.arange(len(regret)),
                                regret-sd_regret,
                                regret+sd_regret,
                                alpha=0.33, linestyle = 'dashed',
                                color = color_codes[agent_name] if color_codes is not None else this_color)
            
            ax_ba.plot(best_action, label = legend_codes[agent_name] + add_string if legend_codes is not None else agent_name + add_string,
                       color = color_codes[agent_name] if color_codes is not None else this_color)
            ax_ba.fill_between(np.arange(len(best_action)),
                                best_action-sd_best_action,
                                best_action+sd_best_action,
                                alpha=0.33, linestyle = 'dashed',
                                color = color_codes[agent_name] if color_codes is not None else this_color)
                            
            ax_ba.set_xlabel('t', fontsize = xy_label_fs)
            
            ax_reg.axvline(x = agent.mean_first_t_opt,
                           color = color_codes[agent_name] if color_codes is not None else this_color)
            
            ax_reg.tick_params(axis='both', labelsize=ticks_fs)
            ax_ba.tick_params(axis='both', labelsize=ticks_fs)
        
        ax_reg.set_title('Online regret', fontsize = titles_fs)
        ax_ba.set_title('% of time best action is chosen', fontsize = titles_fs)

        #ax_reg.legend(loc='lower center', fontsize = 16, bbox_to_anchor=(0.5, 0))
        ax_ba.legend(loc='lower center', fontsize = legend_fs, 
                     bbox_to_anchor=bbox, ncol = ncol if ncol is not None else 1)

        plt.tight_layout(rect=rect)

        fig.savefig(self.foldername + '/regret', dpi = dpi)
        
        plt.show()
    
    def plot_regret_noerr(self, list_agents = None, color_codes = None, 
                    figsize = (12,8), title = None, legend_codes = None,
                    which = 'all', rect = [0, 0.1, 1, 0.9], 
                    bbox = (0.5, -0.25),
                    suptitle_fs = 25, xy_label_fs = 18, 
                    titles_fs = 17.5, ticks_fs = 12.5, legend_fs = 19,
                    ncol = None, dpi = 200):
        '''
        Plots regret and % of best action
        '''
        if list_agents is None:
            list_agents = self.list_agents()
        
        fig    = plt.figure(figsize = figsize)
        ax_reg = fig.add_subplot(211)
        ax_ba  = fig.add_subplot(212)

        fig.suptitle(title, fontsize=suptitle_fs)

        for agent_name in list_agents:
            agent = getattr(self, agent_name)
            
            if which == 'all':
                regret         = agent.mean_regret
                sd_regret      = agent.sd_regret 

                best_action    = agent.mean_best_action
                sd_best_action = agent.sd_best_action

                num = len(agent.success)
            if which == 'success':
                if hasattr(agent, 'mean_regret_s'):
                    regret    = agent.mean_regret_s
                    sd_regret = agent.sd_regret_s

                    best_action    = agent.mean_best_action_s
                    sd_best_action = agent.sd_best_action_s

                    num = len(agent.success)
                else:
                    continue
            if which == 'almost_success':
                if hasattr(agent, 'mean_regret_s_a'):
                    regret    = agent.mean_regret_a_s
                    sd_regret = agent.sd_regret_a_s

                    best_action    = agent.mean_best_action_a_s
                    sd_best_action = agent.sd_best_action_a_s

                    num = len(agent.almost_success)
                else:
                    continue
            
            add_string = ' ({}/{} successful)'.format(num, agent.num_repeats)

            this_color = np.random.rand(3,)

            ax_reg.plot(regret,
                        color = color_codes[agent_name] if color_codes is not None else this_color)
            
            ax_ba.plot(best_action, label = legend_codes[agent_name] + add_string if legend_codes is not None else agent_name + add_string,
                       color = color_codes[agent_name] if color_codes is not None else this_color)
                            
            ax_ba.set_xlabel('t', fontsize = xy_label_fs)
            
            ax_reg.axvline(x = agent.mean_first_t_opt,
                           color = color_codes[agent_name] if color_codes is not None else this_color)
            
            ax_reg.tick_params(axis='both', labelsize=ticks_fs)
            ax_ba.tick_params(axis='both', labelsize=ticks_fs)
        
        ax_reg.set_title('Online regret', fontsize = titles_fs)
        ax_ba.set_title('% of time best action is chosen', fontsize = titles_fs)

        #ax_reg.legend(loc='lower center', fontsize = 16, bbox_to_anchor=(0.5, 0))
        ax_ba.legend(loc='lower center', fontsize = legend_fs, 
                     bbox_to_anchor=bbox, ncol = ncol if ncol is not None else 1)

        plt.tight_layout(rect=rect)

        fig.savefig(self.foldername + '/regret_noerr', dpi = dpi)
        
        plt.show()

    def plot_Q_u(self, list_agents = None, state_actions = None, figsize = (20,20),
                 which = 'all', title = None, colors = None,
                 bbox = (0.5, 0), rect = [0, 0.22, 1, 0.95],
                 suptitle_fs = 25, xy_label_fs = 18, 
                 titles_fs = 17.5, ticks_fs = 12.5, legend_fs = 19, dpi = 200):
        '''
        Plots the evolution of Q and u on plot grids
        for specified state-action pairs or for all if unspecified
        '''
        if list_agents is None:
            list_agents = self.list_agents()
        
        num_states, num_actions = self.env.optimal_Q.shape

        if state_actions is None:
            state_actions = [(i,j) for i in range(num_states) for j in range(num_actions)]
            
            height = num_states
            width  = num_actions

        else:
            assert len(state_actions) % 2 == 0, 'Make it an even number of subplots (state-actions)'

            height = int(len(state_actions)/2)
            width  = height

        figures = {}

        for agent_name in list_agents:

            agent = getattr(self, agent_name)

            #Q = np.array(agent.logger['Q'])
            #u = np.array(agent.logger['u'])

            if which == 'all':
                
                Q  = agent.mean_Qs 
                u  = agent.mean_us

                num = len(agent.success)
            if which == 'success':
                if hasattr(agent, 'mean_regret_s'):
                    
                    Q  = agent.mean_Qs_s
                    u  = agent.mean_us_s

                    num = len(agent.success)
                else:
                    continue
            if which == 'almost_success':
                if hasattr(agent, 'mean_regret_s_a'):
                    
                    Q  = agent.mean_Qs_a_s 
                    u  = agent.mean_us_a_s

                    num = len(agent.almost_success)
                else:
                    continue

            fig, axes = plt.subplots(height, width, figsize=figsize, \
                                     facecolor='w', edgecolor='k')

            fig.subplots_adjust(hspace = .3, wspace=.2)

            add_string = ' ({}/{} successful)'.format(num, agent.num_repeats)
            fig.suptitle(title + add_string if title is not None else agent_name + add_string, fontsize=suptitle_fs)

            axes = axes.ravel()

            this_color = np.random.rand(3,)

            for idx, sa in enumerate(state_actions):
                
                axes[idx].plot(Q[:, sa[0], sa[1]],
                               color = colors[agent_name]['var'] if colors is not None else this_color, 
                               label = r'$\mathbb{E}_{\theta}[Q^{*,\mathcal{W}|\theta_t}]$')
                axes[idx].fill_between(np.arange(Q.shape[0]),
                                       Q[:, sa[0], sa[1]] - 2*np.sqrt(u[:, sa[0], sa[1]]),
                                       Q[:, sa[0], sa[1]] + 2*np.sqrt(u[:, sa[0], sa[1]]),
                                            alpha=0.43, linestyle = '--', 
                                            color = colors[agent_name]['var'] if colors is not None else this_color,
                                            label = r'$\pm 2$Var$_{\theta}[Q^{*,\mathcal{W}|\theta_t}]^{1/2}$' if not np.all(u[:, sa[0], sa[1]]==0) else None)

                axes[idx].set_title('State {}, Action {}'.format(sa[0], sa[1]), fontsize = titles_fs)
                #axes[idx].x_label('Iteration') 

                # Plot optmial Q for reference
                axes[idx].axhline(y=self.env.optimal_Q[sa[0], sa[1]], 
                                  color = 'black', linestyle = '--', 
                                  label = r'$Q^{*,\mathcal{W}}$', alpha = 0.7)
                
                if idx >= (width*(height-1)):
                    axes[idx].set_xlabel('t', fontsize = xy_label_fs)

                if idx % width == 0:
                    axes[idx].set_ylabel(r'$Q^{*,\mathcal{W}|\theta_t}$', fontsize = xy_label_fs) 
                
                axes[idx].tick_params(axis='both', labelsize=ticks_fs)

            handles, labels = axes[idx].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', 
                      fontsize = legend_fs, bbox_to_anchor=bbox,
                      ncol = 3)

            plt.tight_layout(rect = rect)

            figures[agent_name] = fig

            fig.savefig(self.foldername + '/' + agent_name + '_Q', dpi = dpi)

            plt.show()
    
    def plot_Q_u_comparison(self, figsize = (12, 12), color_codes = None, from_t = 0,
                            to_t = None, bbox = (0.5, 0), rect = [0, 0.22, 1, 0.95],
                            suptitle_fs = 25, xy_label_fs = 18, 
                            titles_fs = 17.5, ticks_fs = 12.5, legend_fs = 19, dpi = 200):
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

        if to_t is None:
            to_t = monte_carlo_agent.mean_Qs.shape[0]

        u_mc = monte_carlo_agent.mean_us[from_t:to_t]

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

            Q  = agent.mean_Qs[from_t:to_t] 
            u  = agent.mean_us[from_t:to_t]

            for idx, sa in enumerate(state_actions):
                
                axes[idx].plot(Q[:, sa[0], sa[1]], color = '#3138fb', label = r'$\mathbb{E}_{\theta}[Q^{*,\mathcal{W}|\theta_t}]$')
                axes[idx].fill_between(np.arange(Q.shape[0]),
                                       Q[:, sa[0], sa[1]] - 2*np.sqrt(u[:, sa[0], sa[1]]),
                                       Q[:, sa[0], sa[1]] + 2*np.sqrt(u[:, sa[0], sa[1]]),
                                            alpha=0.43, linestyle = 'None', color = color_codes[agent_name],
                                            label = r'$\pm $2Var$_{\theta}[Q^{*,\mathcal{W}|\theta_t}]^{1/2}$')
                
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
            fig.legend(handles, labels, loc='lower center', fontsize = 16, bbox_to_anchor=bbox)

            plt.tight_layout(rect=rect)

            figures[agent_name] = fig

            fig.savefig(self.foldername + '/' + agent_name + '_Q')

            plt.show()
        
        return figures

    def plot_Q_u_comparison_one_plot(self, figsize = (12, 12), color_codes = None, from_t = 0, to_t = None,
                                     list_agents = None, bbox = (0.5, 0), rect = [0, 0.22, 1, 0.95]):
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

                if to_t is None:
                    to_t = agent.mean_Qs.shape[0]

                Q  = agent.mean_Qs[from_t:to_t] 
                u  = agent.mean_us[from_t:to_t]

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
        fig.legend(handles, labels, loc='lower center', fontsize = 16, bbox_to_anchor=bbox)

        plt.tight_layout(rect=rect)

        fig.savefig(self.foldername + '/' + agent_name + '_Q')

        plt.show()
        
        return fig


    def plot_state_freq(self, list_agents = None, figsize = (12,10),
                        which = 'all', title = None, colors = None,
                        capsize = 5, rect = [0, 0.22, 1, 0.95],
                        suptitle_fs = 25, xy_label_fs = 18, 
                        titles_fs = 17.5, ticks_fs = 12.5, legend_fs = 19, dpi = 200):
        '''
        Implement so that it shows in intervals instead of 10
        '''
        if list_agents is None:
            list_agents = self.list_agents()

        for agent_name in list_agents:

            agent = getattr(self, agent_name)

            #Q = np.array(agent.logger['Q'])
            #u = np.array(agent.logger['u'])

            if which == 'all':
                
                states = agent._env_states
                mean_state_freq = agent.mean_state_freq
                sd_state_freq   = agent.sd_state_freq

                num = len(agent.success)
            if which == 'success':
                if hasattr(agent, 'mean_regret_s'):
                    
                    states = agent._env_states
                    mean_state_freq = agent.mean_state_freq_s
                    sd_state_freq   = agent.sd_state_freq_s

                    num = len(agent.success)
                else:
                    continue
            if which == 'almost_success':
                if hasattr(agent, 'mean_regret_s_a'):
                    
                    states = agent._env_states
                    mean_state_freq = agent.mean_state_freq_a_s
                    sd_state_freq   = agent.sd_state_freq_a_s

                    num = len(agent.almost_success)
                else:
                    continue

            fig, axes = plt.subplots(2, 5, figsize=figsize, \
                                            facecolor='w', edgecolor='k')
            
            add_string = ' ({}/{} successful)'.format(num, agent.num_repeats)
            fig.suptitle(title + add_string if title is not None else agent_name + add_string, fontsize=suptitle_fs)

            axes = axes.ravel()

            agent = getattr(self, agent_name)

            this_color = np.random.rand(3,)

            for i in range(10):
                axes[i].bar(states, mean_state_freq[i, :], yerr=sd_state_freq[i, :], capsize=capsize, 
                            color = colors[agent_name] if colors is not None else this_color, edgecolor = 'black')

                axes[i].set_xticks(states)

                axes[i].set_title('After step = {}'.format((i+1)*int(len(agent.memory_buffer)/10)), fontsize = titles_fs)

                if i >= 5:
                    axes[i].set_xlabel('State', fontsize = xy_label_fs)

                if i % 5 == 0:
                    axes[i].set_ylabel('Visited frequency', fontsize = xy_label_fs) 
                
                axes[i].tick_params(axis='both', labelsize=ticks_fs)
            
            plt.tight_layout(rect=rect)
            
            fig.savefig(self.foldername + '/' + agent_name + '_state_freqs', dpi = dpi)

            plt.show()
    

    def plot_action_freq(self, list_agents = None, figsize = (12,10),
                        which = 'all', title = None, colors = None,
                        capsize = 5, rect = [0, 0.22, 1, 0.95],
                        suptitle_fs = 25, xy_label_fs = 18, 
                        titles_fs = 17.5, ticks_fs = 12.5, legend_fs = 19, dpi = 200):
        '''
        Implement so that it shows in intervals instead of 10
        '''
        if list_agents is None:
            list_agents = self.list_agents()

        for agent_name in list_agents:

            agent = getattr(self, agent_name)

            #Q = np.array(agent.logger['Q'])
            #u = np.array(agent.logger['u'])

            if which == 'all':
                
                states = agent._env_actions
                mean_state_freq = agent.mean_action_freq
                sd_state_freq   = agent.sd_action_freq

                num = len(agent.success)
            if which == 'success':
                if hasattr(agent, 'mean_regret_s'):
                    
                    states = agent._env_states
                    mean_state_freq = agent.mean_state_freq_s
                    sd_state_freq   = agent.sd_state_freq_s

                    num = len(agent.success)
                else:
                    continue
            if which == 'almost_success':
                if hasattr(agent, 'mean_regret_s_a'):
                    
                    states = agent._env_states
                    mean_state_freq = agent.mean_state_freq_a_s
                    sd_state_freq   = agent.sd_state_freq_a_s

                    num = len(agent.almost_success)
                else:
                    continue

            fig, axes = plt.subplots(2, 5, figsize=figsize, \
                                            facecolor='w', edgecolor='k')
            
            add_string = ' ({}/{} successful)'.format(num, agent.num_repeats)
            fig.suptitle(title + add_string if title is not None else agent_name + add_string, fontsize=suptitle_fs)

            axes = axes.ravel()

            agent = getattr(self, agent_name)

            this_color = np.random.rand(3,)

            for i in range(10):
                axes[i].bar(states, mean_state_freq[i, :], yerr=sd_state_freq[i, :], capsize=capsize, 
                            color = colors[agent_name] if colors is not None else this_color, edgecolor = 'black')

                axes[i].set_xticks(states)

                axes[i].set_title('After step = {}'.format((i+1)*int(len(agent.memory_buffer)/10)), fontsize = titles_fs)

                if i >= 5:
                    axes[i].set_xlabel('Action', fontsize = xy_label_fs)

                if i % 5 == 0:
                    axes[i].set_ylabel('Performed frequency', fontsize = xy_label_fs) 
                
                axes[i].tick_params(axis='both', labelsize=ticks_fs)
            
            plt.tight_layout(rect=rect)
            
            fig.savefig(self.foldername + '/' + agent_name + '_action_freqs', dpi = dpi)

            plt.show()
    
        




        


    
    
