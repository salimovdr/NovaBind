import numpy as np
import pandas as pd
import tensorflow as tf
import subprocess as sp

from dna_processing import encode, complement

from sklearn.preprocessing import minmax_scale

from tqdm.auto import trange
import keras.backend as K
import gc

_models = {
    'PBM': [(i, j) for i in range(3) for j in range(3)],
    'HTS': [(0, 0), (2, 2)],
}

_prots = {
    'PBM': ['GCM1', 'MKX', 'MSANTD1', 'MYPOP',
            'SP140L', 'TPRX1', 'ZFTA'],
    'HTS': ['ZNF831', 'ZNF780B', 'ZNF721', 'ZNF500',
            'ZNF286B', 'ZBTB47', 'FIZ1', 'CREB3L3'],
}

_pexp_ws = {
    'GHTS': (58, 9),
    'CHS': (58, 9),
}


def make_test_prediction(model, exp, out_shape):
    sp.run(f'mkdir predict_{exp}', shell=True)
    
    df = pd.read_csv(f'test/{exp}.csv')
    X_test = np.load(f'test/{exp}.npy')
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    for f, s in _models[exp]:
        fold = f'fold{f}'
        seed = f'seed{s}'
        model.load_weights(f'models_{exp}/{fold}_{seed}.keras')

        pred = model.predict(X_test, batch_size=16000)
        pred = pred.reshape(len(df), pred.shape[0]//len(df), out_shape).max(axis=1)

        np.save(f'predict_{exp}/Y_pred_{f}{s}.npy', pred)

    pred = np.load(f'predict_{exp}/Y_pred_00.npy')
    for f, s in _models[exp]:
        pred = pred + np.load(f'predict_{exp}/Y_pred_{f}{s}.npy')
    pred = minmax_scale(pred).round(5)

    df = df.join(pd.DataFrame(pred)).drop('seq', axis=1)
    df.columns = ['id'] + _prots[exp]
    df.to_csv(f'predict_{exp}.tsv', sep='\t', index=False)

    print(f'{exp} prediction are made')


def make_primary_prediction(model, exp, out_shape):
    for pexp in ['GHTS', 'CHS']:
        name = f'predict_{pexp}_on_{exp}'
        sp.run(f'mkdir {name}', shell=True)

        df = pd.read_csv(f'test/{pexp}.csv')
        
        window, stride = _pexp_ws[pexp]
        for f, s in _models[exp]:
            predict = np.empty((0, out_shape), np.float32)

            for i in trange(32):
                X_test = np.load(f'{pexp}_w{window}s{stride}/part_{i}.npy')
                X_test = X_test.astype(np.float32) / 4
                X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

                model.load_weights(f'models_{exp}/fold{f}_seed{s}.keras')

                pred = model.predict(X_test, batch_size=16000, verbose=0)
                predict = np.append(predict, pred, axis=0)

                del X_test, pred
                K.clear_session()
                gc.collect()

            predict = predict.reshape(len(df), predict.shape[0]//len(df), out_shape).max(axis=1)
            np.save(f'{name}/Y_pred_{f}{s}.npy', predict)
        
        predict = np.load(f'{name}/Y_pred_00.npy')
        for f, s in _models[exp][1:]:
            predict = predict + np.load(f'{name}/Y_pred_{f}{s}.npy')
        predict = minmax_scale(predict).round(5)
        
        df = df.join(pd.DataFrame(predict)).drop('seq', axis=1)
        df.columns = ['id'] + _prots[exp]
        df.to_csv(f'{name}.tsv', sep='\t', index=False)

        print(f'{pexp} on {exp} prediction are made')
