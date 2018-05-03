from collections import defaultdict
from typing import Dict, List
import numpy as np


train_file = 'train/sonic-train.csv'


def get_levels() -> Dict[str, List[str]]:
    '''Returns a dictionary of {<game>: [<level>, ...], ...}'''
    import csv

    # levels = defaultdict(lambda:[])
    levels = list()
    with open(train_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # skip header
        for row in reader:
            # levels[row[0]].append(row[1])
            levels.append((row[0], row[1]))
    return levels
    

def split(arr:np.array, axis=0) -> List[np.array]:
    '''Splits a numpy array into its constituent subarrays.'''
    return [np.reshape(a, np.shape(a)[1:]) for a in np.split(arr, arr.shape[axis], axis=axis)]
