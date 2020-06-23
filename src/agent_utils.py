
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
    
    def update(self, which, value):

        if which not in self.memory.keys():
            self.memory[which] = [value]
        else:
            self.memory[which].append(value)


