import numpy as np
from typing import Union


def as_binary_array(x:int, length:Union[None, int]=None) -> np.array:
    length = length if length is not None else x.bit_length()
    b = []
    for _ in range(length):
        b.append(x & 1)
        x = x >> 1
    b.reverse()
    return np.array(b)

def array_as_binary_array(length:int):
    def _aba(x:np.array) -> np.array:
        res = list()
        for xi in x:
           res.append(as_binary_array(xi, length=length))
        return np.array(res)

def as_int(b:np.array) -> int:
    x = 0
    k = len(b)
    for i, bi in zip(range(k-1, 0, -1), b):
        x += bi * (2**i)
    return x

def array_as_int(b:np.array) -> np.array:
    res = list()
    for bi in b:
        res.append(as_int(bi))
    return np.array(res)

def one_hot(values:Union[int, np.array], n_values:int):
    return np.eye(n_values)[values]

def flatten(x:np.array):
    return np.reshape(x, [-1])
