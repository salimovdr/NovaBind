import subprocess as sp
import random as rd
import numpy as np
import pandas as pd
import os

rd.seed(42)
np.random.seed(42)

from prep_utils import (pbm_prots,
                        hts_prots,
                        hts_used_cycles,
                        fasta_to_df)


sp.run('mkdir snp test', shell=True)

sp.run(f'unzip -q data/ibis_rSNP.zip -d snp', shell=True)
print('Archive are extracted')


files = os.listdir('snp')
files = [x for x in files if '@' in x]

df = pd.DataFrame()
for file in files:
    path = f'snp/{file}'
    sdf = fasta_to_df(path)
    sdf['id'] = [f"{file.replace('.fasta', '')}-{i}" for i in sdf.id]
    df = pd.concat([df, sdf], axis=0)

df.to_csv('test/SNP.csv', index=False)

print("Data are converted to csv")

# removing junk files
sp.run('rm -r snp', shell=True)