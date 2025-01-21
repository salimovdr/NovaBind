import subprocess as sp
import random as rd
import numpy as np
import pandas as pd

rd.seed(42)
np.random.seed(42)


pbm_prots = ['GCM1', 'MKX', 'MSANTD1', 'MYPOP',
             'SP140L', 'TPRX1', 'ZFTA']
hts_prots = ['ZNF831', 'ZNF780B', 'ZNF721', 'ZNF500',
             'ZNF286B', 'ZBTB47', 'FIZ1', 'CREB3L3']

hts_used_cycles = {
    'CREB3L3': 'R1_C4',
    'FIZ1': 'R0_C4',
    'ZNF500': 'R0_C3',
    'ZNF780B': 'R0_C3',
    'ZNF831': 'R0_C4',
    'ZNF286B': 'R0_C3',
    'ZBTB47': 'R1_C4',
    'ZNF721': 'R0_C3',
}


def fasta_to_df(path):
    '''Just for the simplest case!'''
    ids = []
    seqs = []
    file = open(path)

    for i, line in enumerate(file):
        line = line.replace('\n', '')

        if not line:
            continue

        if i % 2 == 0:
            ids.append(line.replace('>', '').split(' ')[0])
        else:
            seqs.append(line)

    file.close()
    return pd.DataFrame({'id': ids, 'seq': seqs})
    
