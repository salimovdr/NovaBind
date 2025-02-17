import subprocess as sp
import random as rd
import numpy as np
import pandas as pd
import os

rd.seed(42)
np.random.seed(42)

from prep_utils import (pbm_prots,
                        hts_prots,
                        fasta_to_df)


# make dirs for data and extract .zip with SNP
os.makedirs('snp', exist_ok=True)
os.makedirs('test', exist_ok=True)

sp.run(f'unzip -q data/ibis_rSNP.zip -d snp', shell=True)
print('Archive are extracted')


# get available files and filter 
files = os.listdir('snp')
files = [x for x in files if '@' in x]

targets = pbm_prots + hts_prots
files = [x for x in files if any(prot in x for prot in targets)]


# read .fasta and concat to .csv
df = pd.DataFrame()
for file in files:
    path = f'snp/{file}'
    sdf = fasta_to_df(path)
    
    # column "id" should contain file names
    sdf['id'] = [f"{file.replace('.fasta', '')}-{i}" for i in sdf.id]
    df = pd.concat([df, sdf], axis=0)

df.to_csv('test/SNP.csv', index=False)
print("Data are converted to csv")


# remove junk files
sp.run('rm -r snp', shell=True)
