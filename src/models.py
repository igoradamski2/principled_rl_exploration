import numpy as np
from scipy import special, stats

class BayesianGaussianMixture(object):

    def __init__(self, params):
        '''
        Initializes the Bayesian Mixture Model.
        Input: params -> dict
               params['num_mixes'] -> int (number of gaussian mixtures)
               params['prior_components'] -> list of length num_mixes
               params['prior_alpha'] -> float (prior on alpha)
               params['prior_beta'] -> float (prior on beta)
               params['prior_mu'] -> list of length num_mixes (prior means)
               params['prior_sigma'] -> 
        '''
        self.num_mixes        = params['num_mixes']
        self.prior_components = params['prior_components']
        self.prior_alpha      = params['prior_alpha']
        self.prior_beta       = params['prior_beta']


class BayesianGaussian(object):

    def __init__(self, params):
        '''
        Initializes a Bayesian Gaussian Model
        Input: params -> dict
                params['mu'] -> float (prior on NG location)
                params['lambda'] -> float (prior on NG lambda)
                params['alpha'] -> float (prior on NG alpha)
                params['beta'] -> float (prior on NG beta)
        '''
        self.mu0    = params['mu']
        self.lmbda0 = params['lambda']
        self.alpha0 = params['alpha']
        self.beta0  = params['beta']

        self.mu     = params['mu']
        self.lmbda  = params['lambda']
        self.alpha  = params['alpha']
        self.beta   = params['beta']

    def update(self, data, sufficient = False):
        '''
        Updates the model parameters.
        Input: data -> list/np.array of values or dict of sufficient statistics
               sufficient -> bool (True if data is in form of sufficient statistics) 
        '''
        if not sufficient:
            s_mean, s_var, n = self.get_sufficient_stats(data)
        else:
            s_mean, s_var, n = data['s_mean'], data['s_var'], data['n']
        
        self.mu    = (self.lmbda0*self.mu0 + n*s_mean)/(self.lmbda0 + n)
        self.lmbda = self.lmbda0 + n
        self.alpha = self.alpha0 + 0.5*n
        self.beta  = self.beta0 + 0.5*(n*s_var + (self.lmbda0*n*((s_mean-self.mu0)**2))/(self.lmbda0+n))
    
    def get_predictive_moment(self, which_moment):
        '''
        Returns the predictive moment of the distribution
        '''
        if which_moment == 1:
            # Here we return the predictive mean
            return self.mu
        
        if which_moment == 2:
            # Here we return the second predictive moment
            return (special.gamma(self.alpha-1)/special.gamma(self.alpha))*self.beta
    
    def get_sampled_moments(self, which_moment, num_samples):
        '''
        Returns the moment for a sampled set of parameters
        '''
        # First we need to sample the parameters 
        # a == alpha, scale == 1/beta
        tau = stats.gamma.rvs(a = self.alpha, scale = 1/self.beta, size = num_samples)
        m = self.mu + stats.norm.rvs(size = num_samples)*np.sqrt(1/(self.lmbda*tau))

        # Now get the moments
        if which_moment == 1:
            # Here we return the mean reward given parameters
            return m
        
        if which_moment == 2:
            # Here we return the second moment given parameters
            return (1/tau) + m**2

    def pack_parameters(self):
        pars = {'mu': self.mu,
                'lmbda': self.lmbda,
                'alpha': self.alpha,
                'beta': self.beta
        }
        return pars
    
    @staticmethod    
    def get_sufficient_stats(data):
        ''' 
        Gets the sufficient statistics needed to update this model
        Input: data -> list/np.array of values
        '''
        if type(data) is not np.ndarray:
            data = np.array(data)
        
        s_mean = np.mean(data)
        s_var  = np.var(data)
        n      = len(data)

        return s_mean, s_var, n
    
    @classmethod
    def default(cls):
        params = {'mu': 0,
                  'lambda': 1,
                  'alpha': 2,
                  'beta': 2,
        }
        return cls(params)
    
class BayesianCategorical(object):

    def __init__(self, params):
        '''
        Initializes a Bayesian Categorical Model
        Input: params -> dict
                params['out_size'] -> int (size of output vector, usually number of states)
                params['c'] -> list of floats (prior on c), list of size out_size
        '''
        self.out_size = params['out_size']
        self.c0       = np.array(params['c'])

        self.c        = np.array(params['c'])

    def update(self, data, sufficient = False):
        '''
        Updates the model parameters.
        Input: data -> list/np.array of values or dict of sufficient statistics
               sufficient -> bool (True if data is in form of sufficient statistics) 
        '''
        if not sufficient:
            counts = self.get_sufficient_stats(data)
        else:
            counts = data['counts']
        
        self.c = self.c0 + counts
         
    def get_predictive_moment(self, which_moment):
        '''
        Returns the predictive moment of the distribution
        '''
        
        if which_moment == 1:
            # Return mean (vector)
            return self.c/np.sum(self.c)
        
        if which_moment == 2:
            # Return variance (vector) of multinomial predictive
            normalized = self.c/np.sum(self.c)
            return ((normalized * (1-normalized))/(np.sum(self.c) + 1)) + normalized**2
    
    def get_sampled_moments(self, which_moment, num_samples):
        '''
        Returns the moment for a sampled set of parameters

        Out: probabilities of transitioning to state s' shape: (num_samples, |S|)
        '''
        # First sample from posterior
        kappa = stats.dirichlet.rvs(alpha = self.c, size = num_samples)

        if which_moment == 0:
            return kappa

        if which_moment == 1:
            pass

    def pack_parameters(self):
        pars = {'counts': self.c
        }
        return pars
    
    def get_sufficient_stats(self, data):
        ''' 
        Gets the sufficient statistics needed to update this model
        Input: data -> list/np.array of values
        '''
        if type(data) is not np.ndarray:
            data = np.array(data)
        
        u, c = np.unique(data, return_counts=True)

        counts = np.zeros(self.out_size)

        for idx, c in zip(u,c):
            counts[idx] = c

        return counts
    
    @classmethod
    def default(cls, out_size):
        params = {'out_size': out_size,
                  'c': [0.1]*out_size
        }
        return cls(params)