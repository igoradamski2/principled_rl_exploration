import numpy as np 

class Distribution(object):
    
    def __init__(self):
        pass
    
    @staticmethod
    def normal(*distrib_params):
        return np.random.normal(*distrib_params)

