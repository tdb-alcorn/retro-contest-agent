from agents import all_agents
from typing import Tuple, Iterable, List, Union, TypeVar, ClassVar, Any
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
    # file_regex: ClassVar[Any] = re.compile('[^/]+\.npz$')
    # columns: List[str]= ['states', 'actions', 'rewards', 'next_states', 'dones']
    # save_columns: List[str] = columns[1:]
    file_regex = re.compile('[^/]+\.npz$')
    columns = ['states', 'actions', 'rewards', 'next_states', 'dones']
    save_columns = columns[1:]

    def __init__(self, agent:str, game:str, level:str, episode:int, initial_state:State):
        self.agent = agent
        self.game = game
        self.level = level
        self.episode = episode
        self.initial_state = initial_state
        # self.data:List[Datum] = list()
        self.data = list()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i:int) -> Datum:
        return self.data[i]
    
    def add(self, datum:Datum):
        self.data.append(datum)

    def sample(self, batch_size:int=100, sequential:bool=False) -> List[Datum]:
        num_data = len(self.data)
        if sequential:
            idx = np.random.choice(num_data - batch_size)
            return self.data[idx:idx + batch_size]
        idc = np.random.choice(num_data, size=batch_size, replace=False)
        return [self.data[i] for i in idc]

    def save(self, location:str='.', suffix:str='') -> str:
        '''Returns the name of the file to which the data was saved.'''
        if location[-1] == '/':
            location = location[:-1]
        if suffix != '':
            suffix = '_' + suffix
        save_data = list(zip(*self.data))[1:] + [self.initial_state]
        data = dict(zip(self.save_columns + ['initial_state'], [np.array(d) for d in save_data]))
        filename = '{location}/{agent}_{game}_{level}_{episode}{suffix}.npz'.format(
            location=location,
            suffix=suffix,
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
            agent, game, level, episode, *other = filename.split('_')
        except ValueError:
            raise InvalidEpisodeNameException(filename)
        loaded = np.load(path)
        initial_state = loaded['initial_state']
        data = [utils.split(loaded[name]) for name in cls.save_columns]
        # next_states: List[State] = data[2]
        # states: List[State] = [initial_state] + next_states[:-1]
        next_states = data[2]
        states = [initial_state] + next_states[:-1]
        data = [states] + data
        ep = cls(agent, game, level, episode, initial_state)
        # ep.data: List[Datum] = list(zip(*data))
        ep.data = list(zip(*data))
        return ep


class Memory(object):
    def __init__(self):
        # self.episodes: List[Episode] = list()
        self.episodes = list()
        self.episode_counter = -1
        self.agent = None
        self.game = None
        self.level = None
        self.current_episode = None
        self.array_names = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.dirty = True
    
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
        self.dirty = True

    def add(self, datum:Datum):
        if self.current_episode is None:
            raise Exception("You need to call begin_episode before adding data.")
        self.current_episode.add(datum)
        self.dirty = True
    
    def save(self, location:str='.', suffix:str='') -> List[str]:
        '''Returns the list of filenames that were saved to.'''
        return [episode.save(location=location, suffix=suffix) for episode in self.episodes]
    
    def load(self, filenames:List[str]):
        '''Loads episodes from files on disk.'''
        self.episodes = list()
        for filename in filenames:
            try:
                self.dirty = True
                self.episodes.append(Episode.load(filename))
                self.current_episode = self.episodes[-1]
                self.episode_counter = len(self.episodes)
            except InvalidEpisodeNameException as e:
                print('memory.load: Skipping episode {} because loading it threw an exception:\n\t{}'.format(filename, e))
        
    def sample(self, batch_size:int=100, single_episode:bool=False, **kwargs) -> List[Datum]:
        if single_episode:
            num_episodes = len(self.episodes)
            ep = np.random.choice(num_episodes)
            return self.episodes[ep].sample(batch_size=batch_size, **kwargs)
        if self.dirty:
            # Makes a lookup table for which episode to find a given datum in.
            self.data_index = dict()
            self.num_data = 0
            for ep_idx in range(len(self.episodes)):
                ep_num_data = len(self.episodes[ep_idx])
                for data_idx in range(self.num_data, self.num_data + ep_num_data):
                    self.data_index[data_idx] = (ep_idx, self.num_data)
                self.num_data += ep_num_data
            self.dirty = False
        idc = np.random.choice(self.num_data, size=batch_size, replace=False)
        batch = list()
        for data_idx in idc:
            ep_idx, data_idx_start = self.data_index[data_idx]
            datum = self.episodes[ep_idx].data[data_idx - data_idx_start]
            batch.append(datum)
        return batch
        

def generate():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate training data.")
    parser.add_argument('level', type=int, help='number from 0 to 46 indicating game/level combination')
    parser.add_argument('--save', type=str, dest='save_location', default='.', metavar='path', help='location to save training data.')
    parser.add_argument('--episodes', type=int, dest='num_episodes', default=1, metavar='N', help='number of episodes to run for each agent/game/level combination.')
    parser.add_argument('--save-suffix', type=str, dest='save_suffix', default='', help='optional suffix to add to saved filenames.')

    args = parser.parse_args()

    # memory = Memory()
    all_levels = utils.get_levels()
    # all_levels = {
    #     'SonicTheHedgehog2-Genesis': [
    #         'AquaticRuinZone.Act1',
    #         # 'MetropolisZone.Act1',
    #         # 'MetropolisZone.Act2',
    #     ],
    # }
    for agent_name, agent_constructor in all_agents.items():
        game, level = all_levels[args.level]
        print("Generating data from {game}:{level} with agent {agent}".format(game=game, level=level, agent=agent_name))
        memory = Memory()
        memory.set_meta(agent=agent_name, game=game, level=level)
        train(agent_constructor, args.num_episodes, game=game, state=level, memory=memory, render=False)
        memory.save(location=args.save_location, suffix=args.save_suffix)
        # for game, levels in all_levels.items():
        #     memory = Memory()
        #     for level in levels:
        #         memory.set_meta(agent=agent_name, game=game, level=level)
        #         train(agent_constructor, args.num_episodes, game=game, state=level, memory=memory, render=False)
        #         memory.save(location=args.save_location)

    # TODO We shouldn't save everything all at the end: by this time memory usage has probably outgrown RAM.
    # Instead we should save each episode as they are generated. Then implement memory by simply keeping
    # around a list of episode filenames. When we need to sample them we run to the right spot in disk 
    # and pull it out. ??
    # memory.save()


# def pretrain(self, env, pretrain_length):
#     state = flatten(env.reset())
#     for _ in range(pretrain_length):
#         action = env.action_space.sample()
#         next_state, reward, done, _ = env.step(action)
#         next_state = flatten(next_state)
#         self.memory.add((state, action, reward, next_state, done))
#         state = next_state
#         if done:
#             state = flatten(env.reset())


if __name__ == '__main__':
    generate()
