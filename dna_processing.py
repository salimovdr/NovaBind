import numpy as np
import pandas as pd

_dna_dicts = [
    {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0.25, 0.25, 0.25, 0.25]
    },
    {
        'A': [4, 0, 0, 0],
        'C': [0, 4, 0, 0],
        'G': [0, 0, 4, 0],
        'T': [0, 0, 0, 4],
        'N': [1, 1, 1, 1],
    },
]

def encode(seq, use_int=False):
    if use_int:
        dtype = np.int8
        dna_dict = _dna_dicts[1]
    else:
        dtype = np.float16
        dna_dict = _dna_dicts[0]
    return np.array([dna_dict[base] for base in seq], dtype=dtype)

def complement(seq):
    seq = seq.upper().replace('A', 't'
                              ).replace('T', 'a'
                                        ).replace('G', 'c'
                                                  ).replace('C', 'g')
    return seq.upper()[::-1]


def get_strided(seq, stride, window):
    n = 1 + (len(seq) - window) // stride
    return [seq[i*stride : i*stride+window] for i in range(n)]
    
def flatten(xss):
    return [x for xs in xss for x in xs]