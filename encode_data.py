import os
import numpy as np
import pandas as pd

from dna_processing import (encode,
                            complement)
from sliding_window import cut_sequences

# one-hot encoding of primary discipline test set
# with sliding window sliding (with stride 1 and window 60)
for exp in ['SNP']:
    for s, w in [(1, 60)]:
        cut_sequences(exp, w, s)
        print(f'Slides (window {w}, stide {s}) of {exp} sequences are encoded')
