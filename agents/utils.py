import numpy as np
from typing import Union


def as_binary_array(x:int, length:Union[None, int]=None) -> np.array:
    length = length if length is not None else x.bit_length
    b = []
    for _ in range(length):
        b.append(x & 1)
        x = x >> 1
    b.reverse()
    return np.array(b)

def one_hot(values:Union[int, np.array], n_values:int):
    return np.eye(n_values)[values]

def flatten(x:np.array):
    return np.reshape(x, [-1])
