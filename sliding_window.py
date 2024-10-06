import os
import glob
import subprocess as sp
from multiprocessing import Pool, Manager

import numpy as np
import pandas as pd

from dna_processing import (encode,
                            complement,
                            get_strided,
                            flatten)

def process_chunk(chunk_file, w, s):
    df = pd.read_csv(chunk_file)
    x_test = df['seq']

    x_test = flatten([get_strided(seq, window=w, stride=s) for seq in x_test])

    x_test = (x for seq in x_test for x in (seq, complement(seq)))
    X_test = np.array([encode(seq, use_int=True) for seq in x_test], dtype=np.int8)

    np.save(chunk_file.replace('.csv', '.npy'), X_test)

def cut_sequences(type_exp, w, s):
    folder = f'{type_exp.upper()}_w{w}s{s}'
    os.makedirs(folder, exist_ok=True)
    df = pd.read_csv(f'test/{type_exp}.csv')
    
    chunk_size = len(df) // 32

    for i in range(32):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i < 31 else len(df)
        df_chunk = df[start_index:end_index]
        df_chunk.to_csv(f'{folder}/part_{i}.csv', index=False)
    
    chunk_files = [f'{folder}/part_{i}.csv' for i in range(32)]
    
    with Manager() as manager:
        with Pool(32) as p:
            p.starmap(process_chunk, [(chunk_file, w, s) for chunk_file in chunk_files])
        
        sp.run(f"rm {folder}/*.csv", shell=True)