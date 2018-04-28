from agents import all_agents
from typing import Tuple, Iterable, List, Union, TypeVar
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
        self.episode_counter = 0
        self.agent = None
        self.game = None
        self.level = None
        self.current_episode = None
        self.array_names = ['states', 'actions', 'rewards', 'next_states', 'dones']
    
    def set_meta(self, agent:Union[str, None]=None, game:Union[str, None]=None, level:Union[str, None]=None):
        if agent is not None:
            self.agent = agent
        if game is not None:
            self.game = game
        if level is not None:
            self.level = level

    def begin_episode(self, initial_state:State):
        if self.agent is None or self.game is None or self.level is None:
            raise("You need to call set_meta before beginning an episode.")
        self.episode_counter += 1
        self.current_episode = Episode(self.agent, self.game, self.level, self.episode_counter, initial_state)
        self.episodes.append(self.current_episode)

    def add(self, datum:Datum):
        if self.current_episode is None:
            raise Exception("You need to call begin_episode before adding data.")
        self.current_episode.add(datum)
    
    def save(self) -> List[str]:
        '''Returns the list of filenames that were saved to.'''
        return [episode.save() for episode in self.episodes]


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
                memory.set_meta(agent=agent_name, game=game, level=level)
                train(agent_constructor, 1, game=game, state=level, memory=memory, render=False)

    memory.save()


if __name__ == '__main__':
    generate()
    # state = np.array([0])
    # ep = Episode("agent", "game", "level", 1, state)
    # datum0: Datum = (state, np.array([0]), 1, np.array([1]), False)
    # ep.add(datum0)
    # datum1: Datum = (np.array([1]), np.array([1]), 0, np.array([0]), True)
    # ep.add(datum1)
    # print(ep.data)
    # filename = ep.save()
    # ep2 = Episode.load(filename)
    # print(ep2.data)