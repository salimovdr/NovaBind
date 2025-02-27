import numpy as np
import pandas as pd
from dna_processing import (encode,
                            complement)

for exp in ['SNP']:
    # read target .csv
    df = pd.read_csv(f'test/{exp}.csv')

    # get central 59 bp of sequences
    x_test = (seq[121:-121] for seq in df.seq)
    print('Midle region is extracted')

    # augment by complementary sequences
    x_test = (x for seq in x_test for x in (seq, complement(seq)))
    print('Augmentation by reverse complement are performed')

    # one-hot encode sequences
    X_test = np.array([encode(seq, use_int=True) for seq in x_test], dtype=np.int8)
    print('Sequences are one-hot encoded')

    # save to light int8 array
    np.save(f'test/{exp}.npy', X_test)
