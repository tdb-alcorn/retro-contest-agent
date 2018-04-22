from agents import all_agents
from typing import Tuple, Iterable
import numpy as np
from train.online import train
import train.utils as utils

State = np.array
Action = np.array
Reward = float
Done = bool
Datum = Tuple[State, Action, Reward, State, Done]
Data = Tuple[Iterable[State], Iterable[Action], Iterable[Reward], Iterable[State], Iterable[Done]]

class Memory(object):
    def __init__(self):
        self.data = []
        self.array_names = ['states', 'actions', 'rewards', 'next_states', 'dones']

    def add(self, datum:Datum):
        self.data.append(datum)

    def dump(self) -> Data:
        return zip(*self.data)

    def save(self):
        data = dict(zip(self.array_names, [np.array(d) for d in self.dump()]))
        # TODO don't save the state twice (once as state and once as next_state)
        np.savez_compressed('./data.npz', **data)
    
    def load(self, file='./data.npz'):
        loaded = np.load(file)
        data = [utils.split(loaded[name]) for name in self.array_names]
        self.data = zip(*data)



def generate():
    memory = Memory()
    # all_levels = utils.get_levels()
    all_levels = {
        'SonicTheHedgehog2-Genesis': [
            'AquaticRuinZone.Act1',
        ],
    }
    for agent_name, agent_constructor in all_agents.items():
        print("Generating data from {}".format(agent_name))
        for game, levels in all_levels.items():
            for level in levels:
                train(agent_constructor, 1, game=game, state=level, memory=memory, render=False)

    memory.save()


if __name__ == '__main__':
    generate()