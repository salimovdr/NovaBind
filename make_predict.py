import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
from tqdm.auto import trange

from data_reading import read_dataset
from architecture import build_model
from predict_utils import make_primary_prediction
from keras.losses import CategoricalCrossentropy

from backend import set_seed, set_device
set_seed(42), set_device(device)



# read argument from command line
parser = argparse.ArgumentParser()
parser.add_argument('--type_exp', type=str, required=True)
parser.add_argument('--device', type=str, required=True)
args = parser.parse_args()
exp = args.type_exp
device = args.device


# initialization
if exp == 'PBM':
    out_shape = 7
    loss = 'mse'
else:
    out_shape = 8
    loss = CategoricalCrossentropy(from_logits=True)
    
model = build_model(loss, out_shape)


X, Y = read_dataset(exp, f'folds_{exp}/fold0', small=True)
_ = model.fit(X, Y,
              batch_size=16000,
              epochs=1,
              verbose=0)

# prediction
make_primary_prediction(model, exp, out_shape)
