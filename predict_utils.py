import os
import numpy as np
import pandas as pd
import tensorflow as tf

from dna_processing import encode, complement

from sklearn.preprocessing import minmax_scale
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

def make_snp_prediction(model, exp, out_shape):
    for pexp in ['SNP']:
        name = f'predict_{pexp}_on_{exp}'
        os.makedirs(name, exist_ok=True)
        
        # load test table with correct sequence id
        df = pd.read_csv(f'test/{pexp}.csv')

        # reading encoded sequences
        X_test = np.load(f'test/{pexp}.npy')
        X_test = X_test.astype(np.float32) / 4
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        
        # iter by each model in ensemble 
        for f, s in _models[exp]:
            predict = np.empty((0, out_shape), np.float32)

            model.load_weights(f'models_{exp}/fold{f}_seed{s}.keras')

            pred = model.predict(X_test, batch_size=16000, verbose=1)
            predict = np.append(predict, pred, axis=0)

            # clear memory
            del pred
            K.clear_session()
            gc.collect()

            # take a maximum from complement sequence
            assert predict.shape[0]//len(df) == 2
            
            predict = predict.reshape(len(df), 2, out_shape).max(axis=1)
            np.save(f'{name}/Y_pred_{f}{s}.npy', predict)

        # load prediction of each model and average them
        predict = np.load(f'{name}/Y_pred_00.npy')
        for f, s in _models[exp][1:]:
            predict = predict + np.load(f'{name}/Y_pred_{f}{s}.npy')
        predict = minmax_scale(predict).round(6)


        # join prediction with sequence id and save
        df = df.join(pd.DataFrame(predict)).drop('seq', axis=1)
        df.columns = ['id'] + _prots[exp]
        df.to_csv(f'{name}.tsv', sep='\t', index=False)

        print(f'{pexp} on {exp} prediction are made')
