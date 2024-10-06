import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture

def gauss_transform(Y_train, Y_val):
    y_train = Y_train.reshape(-1, 1)
    y_val = Y_val.reshape(-1, 1)
    
    gmm = GaussianMixture(2)    
    gmm.fit(y_train)
    
    means = gmm.means_.flatten()
    index = np.argmax(means)
    
    proba = gmm.predict_proba(y_train)[:, index]
    y_train = np.array([(p * s)[0] for p, s in zip(proba, y_train)])

    proba = gmm.predict_proba(y_val)[:, index]
    y_val = np.array([(p * s)[0] for p, s in zip(proba, y_val)])
    
    Y_train = y_train.reshape(Y_train.shape)
    Y_val = y_val.reshape(Y_val.shape)
    return Y_train, Y_val

def convert(array):
    return tf.convert_to_tensor(array, dtype=tf.float32)

def read_dataset(type_exp, path='.', full=False, small=False):
    subsets = ['X_train', 'X_val', 'Y_train', 'Y_val']

    X_train, X_val, Y_train, Y_val = [np.load(f'{path}/{name}.npy') for name in subsets]

    if type_exp == 'PBM':
        Y_train, Y_val = gauss_transform(Y_train, Y_val)

    if full:
        X = np.concatenate([X_train, X_val])
        Y = np.concatenate([Y_train, Y_val])
        XY = X, Y
    elif small:
        XY = X_val, Y_val
    else:
        XY = X_train, X_val, Y_train, Y_val

    XY = [convert(arr) for arr in XY]
    return XY
