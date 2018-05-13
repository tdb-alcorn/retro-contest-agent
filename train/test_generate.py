import numpy as np
import time
import glob
from train.generate import Memory, Episode, Datum


def test_memory():
    def time_one(fn, name=None):
        name = 'Operation' if name is None else name
        start = time.time()
        res = fn()
        end = time.time()
        print('%s took %.2fs' % (name, (end - start)))
        return res

    files = glob.glob('./**/*_*_*_*.npz', recursive=True)
    print(files)
    memory = Memory()
    time_one(lambda:memory.load(files), 'memory.load')
    print(time_one(lambda:memory.sample(batch_size=100), name='memory.sample')[0])
    print(time_one(lambda:memory.sample(batch_size=100), name='memory.sample')[0])

# def test_episode():
#     state = np.array([0])
#     ep = Episode("agent", "game", "level", 1, state)
#     datum0: Datum = (state, np.array([0]), 1, np.array([1]), False)
#     ep.add(datum0)
#     datum1: Datum = (np.array([1]), np.array([1]), 0, np.array([0]), True)
#     ep.add(datum1)
#     print(ep.data)
#     filename = ep.save()
#     ep2 = Episode.load(filename)
#     print(ep2.data)