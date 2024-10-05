import subprocess as sp
import random as rd
import numpy as np
import pandas as pd

rd.seed(42)
np.random.seed(42)

from utils import (pbm_prots,
                   hts_prots,
                   hts_used_cycles,
                   fq2df,
                   merge_pbm,
                   merge_hts,
                   k_fold_split,
                   )


sp.run('mkdir pbm hts test', shell=True)


# Экстракция архивов
for dset in ['train', 'test']:
    sp.run(f'unzip -q data/IBIS.{dset}_data.Final.v1.zip', shell=True)

for prot in pbm_prots:
    folder = f'train/PBM/{prot}'
    sp.run(f'mv {folder}/QNZS_*.tsv pbm/{prot}.tsv', shell=True)

for prot, cycle in hts_used_cycles.items():
    folder = f'train/HTS/{prot}'
    sp.run(f'mv {folder}/{prot}_{cycle}_*.fastq.gz hts/{prot}.fq.gz',
           shell=True)

for exp in ['PBM', 'HTS', 'GHTS', 'CHS']:
    sp.run(f'mv {exp}_*.fasta test/{exp}.fna', shell=True)


# Компиляция данных PBM и разбиение на фолды
df = merge_pbm()
k_fold_split(df, exp='pbm')


# Компиляция данных HTS
sp.run('unpigz -f hts/*.gz', shell=True)
df = merge_hts()
k_fold_split(df, exp='hts')

# Удаление лишних файлов
sp.run('rm -r train pbm hts *.fasta *.bed', shell=True)


