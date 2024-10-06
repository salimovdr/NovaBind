import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from backend import set_seed, set_device

from data_reading import read_dataset
from architecture import build_model

from keras.callbacks import EarlyStopping
from architecture import CosineAnnealingScheduler, TQDMProgressBar

from keras.losses import CategoricalCrossentropy

f = 0
fold = f'fold{f}'

s = 0
seed = f'seed{s}'

device = 0

set_seed(s)
#set_device(device)

epochs = 500
patience = 60
batch_size = 983


X_train, X_val, Y_train, Y_val = read_dataset(path=f'folds_HTS/{fold}', type_exp='HTS')
model = build_model(output_shape=8, loss=CategoricalCrossentropy(from_logits=True))

callbacks = [
    CosineAnnealingScheduler(),
    EarlyStopping(patience=patience, restore_best_weights=True),
    TQDMProgressBar(epochs),
]


history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=0)

if not os.path.exists('models_HTS'):
    os.makedirs('models_HTS')
    
model.save_weights(f'models_HTS/{fold}_{seed}.keras')