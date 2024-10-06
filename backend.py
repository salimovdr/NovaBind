import os
import random as rn
import numpy as np
import tensorflow as tf

def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def set_device(device=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
