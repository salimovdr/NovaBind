import os
import numpy as np
import pandas as pd

from dna_processing import (encode,
                            complement)
from sliding_window import cut_sequences


def encode_folds(path, n_folds=3):
    dfs = [pd.read_csv(f'{path}/fold_{i}.csv') for i in range(n_folds)]

    for i in range(n_folds):
        train = pd.concat(dfs[:i] + dfs[i+1:])
        val = dfs[i]
    
        for subset, df in zip(['train', 'val'], [train, val]):
            os.makedirs(f'{path}/fold{i}', exist_ok=True)
            
            X = df.seq
            X = (x for seq in X for x in (seq, complement(seq)))
            X = np.array([encode(seq) for seq in X],
                         dtype=np.float16)
            np.save(f'{path}/fold{i}/X_{subset}.npy', X)
    
            Y = df.iloc[:, 1:].to_numpy()
            Y = np.array([target for target in Y for _ in (0, 1)],
                         dtype=np.float32)
            np.save(f'{path}/fold{i}/Y_{subset}.npy', Y)

# one-hot encoding of train set folds
for exp in ['PBM', 'HTS']:
    path = f'folds_{exp}'
    encode_folds(path)
    print(f'{exp} folds are compiled and encoded')


# one-hot encoding of seconday disciplines test set
for exp in ['PBM', 'HTS']:
    df = pd.read_csv(f'test/{exp}.csv')
    X = df.seq
    X = (x for seq in X for x in (seq, complement(seq)))
    X = np.array([encode(seq) for seq in X], dtype=np.float16)
    np.save(f'test/{exp}.npy', X)
    print(f'{exp} test set are encoded')

# one-hot encoding of primary discipline test set
# with sliding window sliding (with strides 1 and 9)
for exp in ['GHTS', 'CHS']:
    for s, w in [(9, 58)]:
        cut_sequences(exp, w, s)
        print(f'Slides (window {w}, stide {s}) of {exp} sequences are encoded')
