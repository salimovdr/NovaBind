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


def combine(path, type_exp, w, s, count_chunks=32):
    combined_data = []
    
    for i in range(32):
        file_path = f'{path}/part_{i}.npy'
        data_part = np.load(file_path)
        combined_data.append(data_part)
    final_data = np.concatenate(combined_data)
    
    np.save(f'test/{type_exp}_w{w}s{s}_combined.npy', final_data)


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
        combine(path=folder, type_exp=type_exp, w=w, s=s)
        
        sp.run(f"rm -r {folder}", shell=True)