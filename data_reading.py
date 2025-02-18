import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture

def _gauss_transform(Y_train, Y_val):
    """Apply GaussianMixture transformation to train and val subset by train subset.
       Needed for training on PBM data.
    """
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

def _convert(array):
    """Convert any collection to tensorflow float32 tensor."""
    return tf.convert_to_tensor(array, dtype=tf.float32)


def read_dataset(type_exp, path='.', full=False, small=False):
    """Load and convert early prepared fold.
       Apply transformation if needed.
    """ 
    subsets = ['X_train', 'X_val', 'Y_train', 'Y_val']

    X_train, X_val, Y_train, Y_val = [np.load(f'{path}/{name}.npy') for name in subsets]

    if type_exp == 'PBM':
        Y_train, Y_val = _gauss_transform(Y_train, Y_val)

    if full:
        # if you want not splitted data
        X = np.concatenate([X_train, X_val])
        Y = np.concatenate([Y_train, Y_val])
        XY = X, Y
    elif small:
        # for initialization
        XY = X_val, Y_val
    else:
        # defulat
        XY = X_train, X_val, Y_train, Y_val

    XY = [_convert(arr) for arr in XY]
    return XY
