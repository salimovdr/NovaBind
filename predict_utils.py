import os
import numpy as np
import pandas as pd
import tensorflow as tf

from dna_processing import encode, complement

from sklearn.preprocessing import minmax_scale

from tqdm.auto import trange
import keras.backend as K
import gc

# how many models (by seed and folds) we have in ensemble for each train subset?
_models = {
    'PBM': [(i, j) for i in range(3) for j in range(3)],
    'HTS': [(i, j) for i in range(3) for j in range(3)],
}

# which proteins we predict based on each train subset?
_prots = {
    'PBM': ['GCM1', 'MKX', 'MSANTD1', 'MYPOP',
            'SP140L', 'TPRX1', 'ZFTA'],
    'HTS': ['ZNF831', 'ZNF780B', 'ZNF721', 'ZNF500',
            'ZNF286B', 'ZBTB47', 'FIZ1', 'CREB3L3'],
}

# how we slice sequences from each target subset?
_pexp_ws = {
    'SNP': (60, 1),
}


def make_primary_prediction(model, exp, out_shape):
    for pexp in ['SNP']:
        name = f'predict_{pexp}_on_{exp}'
        os.makedirs(name, exist_ok=True)
        
        window, stride = _pexp_ws[pexp]
        
        # iter by each model in ensemble 
        for f, s in _models[exp]:
            predict = np.empty((0, out_shape), np.float32)

            # test set had been split to 64 subset due to memory limits
            for i in trange(64):
                X_test = np.load(f'{pexp}_w{window}s{stride}/part_{i}.npy')
                X_test = X_test.astype(np.float32) / 4
                X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

                model.load_weights(f'models_{exp}/fold{f}_seed{s}.keras')

                pred = model.predict(X_test, batch_size=10000, verbose=0)
                predict = np.append(predict, pred, axis=0)

                # clear memory
                del X_test, pred
                K.clear_session()
                gc.collect()

            # take a maximum from complement sequence and slices
            predict = predict.reshape(len(df), predict.shape[0]//len(df), out_shape).max(axis=1)
            np.save(f'{name}/Y_pred_{f}{s}.npy', predict)

        # load prediction of each model and average them
        predict = np.load(f'{name}/Y_pred_00.npy')
        for f, s in _models[exp][1:]:
            predict = predict + np.load(f'{name}/Y_pred_{f}{s}.npy')
        predict = minmax_scale(predict).round(5)

        # load test table with correct sequence id and join with them
        df = pd.read_csv(f'test/{pexp}.csv')
        df = df.join(pd.DataFrame(predict)).drop('seq', axis=1)
        df.columns = ['id'] + _prots[exp]
        df.to_csv(f'{name}.tsv', sep='\t', index=False)

        print(f'{pexp} on {exp} prediction are made')
