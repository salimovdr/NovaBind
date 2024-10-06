import numpy as np
import pandas as pd
import subprocess as sp

df1 = pd.read_csv('predict_GHTS_on_PBM.tsv', index_col=0, sep='\t')
df2 = pd.read_csv('predict_GHTS_on_HTS.tsv', index_col=0, sep='\t')
df = df1.join(df2)
df.to_csv('predict_GHTS.tsv', sep='\t')

df1 = pd.read_csv('predict_CHS_on_PBM.tsv', index_col=0, sep='\t')
df2 = pd.read_csv('predict_CHS_on_HTS.tsv', index_col=0, sep='\t')
df = df1.join(df2)
df.to_csv('predict_CHS.tsv', sep='\t')

sp.run('rm predict_*_on_*.tsv', shell=True)

sp.run('pigz --best *.tsv')