import numpy as np


class DecayProcess(object):
    def __init__(self, explore_start:float=1.0, explore_stop:float=1e-2, decay_rate:float=1e-4):
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.explore_range = self.explore_start - self.explore_stop
        self.counter = 0
        
    def step(self):
        self.counter += 1
    
    def sample(self, size=None):
        epsilon = self.explore_stop + self.explore_range * np.exp(-self.decay_rate * self.counter)
        return (epsilon > np.random.random(size=size)).astype(np.int32)
    
    def reset(self):
        self.counter = 0