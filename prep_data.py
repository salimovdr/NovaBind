import subprocess as sp
import random as rd
import numpy as np
import pandas as pd

rd.seed(42)
np.random.seed(42)

from prep_utils import (pbm_prots,
                        hts_prots,
                        hts_used_cycles,
                        fq2df,
                        merge_pbm,
                        merge_hts,
                        fasta_to_df,
                        k_fold_split)


sp.run('mkdir pbm hts test', shell=True)

# zip's extraction
for dset in ['train', 'test']:
    sp.run(f'unzip -q data/IBIS.{dset}_data.Final.v1.zip', shell=True)
print('Archives are extracted')

for prot in pbm_prots:
    folder = f'train/PBM/{prot}'
    sp.run(f'mv {folder}/QNZS_*.tsv pbm/{prot}.tsv', shell=True)
print('PBM train data are converted to csv')

for prot, cycle in hts_used_cycles.items():
    folder = f'train/HTS/{prot}'
    sp.run(f'mv {folder}/{prot}_{cycle}_*.fastq.gz hts/{prot}.fq.gz',
           shell=True)
print('HTS train data are converted to csv')

for exp in ['PBM', 'HTS', 'GHTS', 'CHS']:
    df = fasta_to_df(f'{exp}_participants.fasta')
    df.to_csv(f'test/{exp}.csv', index=False)
print("All test data are converted to csv's")

# merge PBM's and split to folds
df = merge_pbm()
k_fold_split(df, exp='pbm')
print('PBM are splitted to folds')

# merge HTS's and split to folds
sp.run('unpigz -f hts/*.gz', shell=True)
df = merge_hts().sample(frac=1).reset_index(drop=True)
k_fold_split(df, exp='hts')
print('HTS are splitted to folds')

# removing junk files
sp.run('rm -r train pbm hts *.fasta *.bed', shell=True)