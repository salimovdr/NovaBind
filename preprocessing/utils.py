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


def fq2df(file_name):
    seqs = []

    with open(file_name) as file:
        while True:

            id = file.readline().strip()
            if not id:
                break

            seq = file.readline().strip()

            file.readline()
            qual = file.readline().strip()

            seqs.append(seq)

    df = pd.DataFrame({'seq': seqs})
    return df


def merge_pbm(prots=pbm_prots):
    prot = prots[0]
    df = pd.read_csv(f'pbm/{prot}.tsv', sep='\t')

    df['seq'] = df.linker_sequence + df.pbm_sequence
    df = df[['seq', 'mean_signal_intensity']]
    df.columns = ['seq', 'GCM1']

    for prot in prots[1:]:
        signal = pd.read_csv(f'pbm/{prot}.tsv', sep='\t')['mean_signal_intensity']
        df[prot] = signal

    return df


def merge_hts(prots=hts_prots):
    df = pd.DataFrame()

    for prot in prots:
        sdf = fq2df(f'hts/{prot}.fq').drop_duplicates('seq')
        sdf[prot] = 1
        df = pd.concat([df, sdf])

    df = df.drop_duplicates('seq')
    df = df.fillna(0)

    return df


def k_fold_split(df, exp, n_folds=3):
    sp.run(f'mkdir {exp}_folds', shell=True)
    
    size = len(df) // n_folds + (1 if len(df) % n_folds != 0 else 0)

    for i in range(n_folds):
        ifold = df.iloc[i*size : min((i+1)*size, len(df))]
        ifold.to_csv(f'{exp}_folds/fold_{i}.csv', index=False)
    
