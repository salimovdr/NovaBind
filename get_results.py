import os
import numpy as np
import pandas as pd
import subprocess as sp

prots = ['GCM1', 'MKX', 'MSANTD1', 'MYPOP', 'SP140L', 'TPRX1', 'ZFTA',
         'ZNF831', 'ZNF780B', 'ZNF721', 'ZNF500', 'ZNF286B', 'ZBTB47',
         'FIZ1', 'CREB3L3']


df1 = pd.read_csv('predict_SNP_on_PBM.tsv', index_col=0, sep='\t')
df2 = pd.read_csv('predict_SNP_on_HTS.tsv', index_col=0, sep='\t')
df = df1.join(df2)


df = df.reset_index()
df['exp'] = df.id.map(lambda x: x.split('-')[0].split('@')[0])
df['prot'] = df.id.map(lambda x: x.split('-')[0].split('@')[1])
df['pos'] = df.id.map(lambda x: x.split('-')[1])



df = df[df.prot.isin(prots)]
df['score'] = df.apply(lambda x: x[x['prot']], axis=1)
df = df[['exp', 'prot', 'pos', 'score']]


folder = 'SalFroLabSNP'
os.makedirs(folder, exist_ok=True)

for (exp, prot), sdf in df.groupby(['exp', 'prot']):
    sdf.score.to_csv(f'{folder}/{exp}@{prot}.txt', index=False, header=False)
    
df.to_csv(f'{folder}/concated.tsv', sep='\t', index=False)
sp.run(f'zip -qr {folder} {folder}', shell=True)
sp.run(f'rm -r {folder}', shell=True)
