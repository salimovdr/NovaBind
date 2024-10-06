import numpy as np

from keras.models import Model, Sequential
from keras.layers import Conv1D, Dense
from keras.layers import GRU, Bidirectional, Concatenate
from keras.layers import GlobalMaxPooling1D, BatchNormalization, Dropout

from keras.callbacks import Callback, EarlyStopping
from keras.optimizers import AdamW

from tqdm.auto import tqdm
import keras.backend as K

from keras.losses import CategoricalCrossentropy

class CosineAnnealingScheduler(Callback):
    def __init__(self, steps=10, lr_max=1e-2, lr_min=1e-5):
        super(CosineAnnealingScheduler, self).__init__()
        self.steps = steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.semigap = (lr_max - lr_min) / 2
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.lr_min + self.semigap*(1 + np.cos(np.pi*(0.5 + epoch/self.steps)))
        K.set_value(self.model.optimizer.lr, lr)

class TQDMProgressBar(Callback):
    def __init__(self, epochs):
        super(TQDMProgressBar, self).__init__()
        self.epochs = epochs
        self.tqdm_bar = None
    def on_train_begin(self, logs=None):
        self.tqdm_bar = tqdm(total=self.epochs)
    def on_epoch_end(self, epoch, logs=None):
        self.tqdm_bar.update(1)
        print(f"\r    Train loss: {logs['loss']:.5f}   Validation loss: {logs['val_loss']:.5f}", end='')
    def on_train_end(self, logs=None):
        self.tqdm_bar.close()


class NeuralNet(Model):
    def __init__(
        self,
        output_shape,
        kernel_a=7, kernel_b=15,
        filters=80, filters_del=4,
        gru_units=325,
        dens_units=500, dens_del=2,
        ):
        super().__init__()
        self.conv_a = Conv1D(kernel_size=kernel_a, filters=filters,
                             use_bias=False,
                             padding='same',
                             activation='relu')
        self.conv_b = Conv1D(kernel_size=kernel_b, filters=filters//filters_del,
                             use_bias=False,
                             padding='same',
                             activation='relu')
        self.concat_i = Concatenate()
        
        self.bigru = Bidirectional(
            GRU(units=gru_units, return_sequences=True),
            merge_mode='sum',
        )
        self.grunorm = BatchNormalization()
        self.concat_j = Concatenate()
        
        self.mpool = GlobalMaxPooling1D()
        self.dens_i = Dense(units=dens_units,
                            activation='silu')
        self.dens_j = Dense(units=dens_units//dens_del,
                            activation='silu')
        self.lin = Dense(units=output_shape)
    
    def call(self, x):
        a, b = self.conv_a(x), self.conv_b(x)
        xab = self.concat_i([x, a, b])
        c = self.grunorm(self.bigru(xab))
        abc = self.concat_j([a, b, c])
        z = self.dens_i(self.mpool(abc))
        y = self.lin(self.dens_j(z))
        return y

def build_model(loss, output_shape, lr=0.01, wd=0.45):
    model = NeuralNet(output_shape=output_shape)
    opt = AdamW(learning_rate=lr,
                weight_decay=wd)
    model.compile(optimizer=opt, loss=loss)
    return model

