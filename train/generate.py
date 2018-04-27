from agents import all_agents
from typing import Tuple, Iterable, List
import numpy as np
from train.online import train
import train.utils as utils
import re

State = np.array
Action = np.array
Reward = float
Done = bool
Datum = Tuple[State, Action, Reward, State, Done]
Data = Tuple[Iterable[State], Iterable[Action], Iterable[Reward], Iterable[State], Iterable[Done]]

class InvalidEpisodeNameException(Exception):
    def __init__(self, filename:str):
        msg = "Invalid episode file name: {}\nShould be of form <agent>_<game>_<level>_<episode>.npz"
        super().__init__(msg.format(filename))

class Episode(object):
    file_regex = re.compile('[^/]+\.npz$')
    columns = ['states', 'actions', 'rewards', 'next_states', 'dones']
    save_columns = columns[1:]

    def __init__(self, agent:str, game:str, level:str, episode:int, initial_state:State):
        self.agent = agent
        self.game = game
        self.level = level
        self.episode = episode
        self.initial_state = initial_state
        self.data = list()

    def add(self, datum:Datum):
        self.data.append(datum)

    def save(self) -> str:
        '''Returns the name of the file to which the data was saved.'''
        save_data = list(zip(*self.data))[1:] + [self.initial_state]
        data = dict(zip(self.save_columns + ['initial_state'], [np.array(d) for d in save_data]))
        filename = '{agent}_{game}_{level}_{episode}.npz'.format(
            agent=self.agent,
            game=self.game,
            level=self.level,
            episode=self.episode)
        np.savez_compressed(filename, **data)
        return filename

    @classmethod
    def load(cls, path:str):
        match = cls.file_regex.search(path)
        if match is None:
            raise InvalidEpisodeNameException(path)
        idx = match.span()
        filename = match.string[idx[0]:idx[1]]
        filename = filename[:-4]  # remove .npz
        try:
            agent, game, level, episode = filename.split('_')
        except ValueError:
            raise InvalidEpisodeNameException(filename)
        loaded = np.load(path)
        initial_state = loaded['initial_state']
        data = [utils.split(loaded[name]) for name in cls.save_columns]
        next_states: List[State] = data[2]
        states: List[State] = [initial_state] + next_states[:-1]
        data = [states] + data
        ep = cls(agent, game, level, episode, initial_state)
        ep.data: List[Datum] = list(zip(*data))
        return ep



class Memory(object):
    def __init__(self):
        self.episodes: List[Episode] = list()
        self.data = []
        self.array_names = ['states', 'actions', 'rewards', 'next_states', 'dones']

    def

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
    # generate()
    state = np.array([0])
    ep = Episode("agent", "game", "level", 1, state)
    datum0: Datum = (state, np.array([0]), 1, np.array([1]), False)
    ep.add(datum0)
    datum1: Datum = (np.array([1]), np.array([1]), 0, np.array([0]), True)
    ep.add(datum1)
    print(ep.data)
    filename = ep.save()
    ep2 = Episode.load(filename)
    print(ep2.data)