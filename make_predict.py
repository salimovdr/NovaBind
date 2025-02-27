import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
from tqdm.auto import trange

from architecture import build_model
from predict_utils import make_snp_prediction
from keras.losses import CategoricalCrossentropy
from backend import set_seed, set_device

# read argument from command line
parser = argparse.ArgumentParser()
parser.add_argument('--type_exp', type=str, required=True)
parser.add_argument('--device', type=str, required=True)
args = parser.parse_args()
exp = args.type_exp
device = args.device

# set constants
set_seed(42), set_device(device)


# which loss we will use?
if exp == 'PBM':
    out_shape = 7
    loss = 'mse'
else:
    out_shape = 8
    loss = CategoricalCrossentropy(from_logits=True)

# init model
model = build_model(loss, out_shape)

# predict by averaging ensemble
make_snp_prediction(model, exp, out_shape)
