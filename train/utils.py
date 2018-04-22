import csv
from collections import defaultdict
from typing import Dict, List


train_file = 'sonic-train.csv'


def get_levels() -> Dict[str, List[str]]:
    levels = defaultdict(lambda:[])
    with open(train_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            levels[row[0]].append(row[1])
    return levels



# TODO: Define a wrapper for retro.env that implements __enter__ and __exit__
# so that I can write `with make(...) as env:`