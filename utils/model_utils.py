import numpy as np
from typing import List


def chunk_as_size(lst, chunk_lst):
    sizes = [len(x) for x in chunk_lst]
    assert sum(sizes) == len(lst)

    rnt = []
    i = 0
    for s in sizes:
        rnt.append(lst[i : i + s])
        i += s

    return rnt


def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def chunks_by_num(lst, n):
    rnt = np.array_split(lst, n)
    return [x.tolist() for x in rnt]


def flatten_list(lst):
    new_lst = []
    for item in lst:
        new_lst += item
    return new_lst
