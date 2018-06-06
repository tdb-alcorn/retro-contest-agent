import csv
import os
from collections import defaultdict
from typing import Dict, List


curdir = os.path.dirname(__file__)
train_file = os.path.join(curdir, 'sonic-train.csv')
validation_file = os.path.join(curdir, 'sonic-validation.csv')


def get_levels_by_game() -> Dict[str, List[str]]:
    '''Returns a dictionary of {<game>: [<level>, ...], ...}'''
    levels = defaultdict(lambda:[])
    for filename in [train_file, validation_file]:
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)  # skip header
            for row in reader:
                levels[row[0]].append(row[1])
    return levels