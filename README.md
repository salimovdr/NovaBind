# NovaBind
Salimov and Frolov Laboratory — participants in the international [Ibis](https://ibis.autosome.org) competition for predicting transcription factor binding levels to DNA sequences. We took part in predicting on genome sequences, based on artificial data.  More details about the architecture and methods used can be found in [google-document](https://clck.ru/3Ddv7i).

Here we demonstrate the performance of NovaBind as we predicted the data in the competition. Due to time constraints, you may notice some 'oddities', such as a different number of models in the ensemble for various TFs. This will be addressed in future versions of <b>NovaBind</b>.

## Environment
We used a server with a GPU running Ubuntu 20.04.6. To set up the environment, please use the following command:

```
conda env create -f environment.yml
```

## Input data
You can find the input data on the [Ibis site](https://ibis.autosome.org/download_data/final). The archive is too large, so we are not attaching it here. However, we have prepared test data to verify the correct execution of <b>NovaBind</b> training and prediction. Please download the archive, unzip it, and place the `data` folder into the root folder where all the scripts from the repository are located.

## Reproduction

### Data preprocessing
**Step 1.** It is necessary to extract the files and convert them to a unified .csv format. For future model ensembling, we will immediately split the data into folds. All unnecessary and temporary files are deleted, leaving only the `folds_PBM` and `folds_HTS` directories with the necessary data. 

To run the script that does this, execute the following command in bash:

```
python prep_data.py
```

**Step 2.** We are ready to split the folds into training and validation sets. DNA sequences are encoded using one-hot encoding, with complementary sequences added to the data. For the data from the GHTS and CHS experiments, sequence segmentation is performed using a sliding window (with strides of 1 and 9). These actions are performed in the `encode_data.py` script:

```
python encode_data.py
```

### Training

**Step 3.** To start training, use the script `parallel_training.py`. Depending on the type of experiment, the training process differs slightly: for PBM experiments, three repetitions with seed = 0, 1, and 2 are run for each of the three folds. For the HTS experiment, training was conducted on the 0th fold with seed = 0 and on the 2nd fold with seed = 2. To choose the training mode, specify the argument --type_exp, which can take the values 'PBM' or 'HTS'. Note that training runs in parallel on the available graphics cards. In the case of training on PBM experiment data, if fewer than 9 graphics cards are available, all available devices will be used, and the tasks will be queued. For the HTS experiment, 2 graphics cards will be used (or a queue of two tasks will be created if only one graphics card is available).

We recommend running the following two commands in sequence, with the second one delayed until the first training stage is complete.

```
python parallel_training.py --type_exp PBM
```
```
python parallel_training.py --type_exp HTS
```

The model weights are saved in the `models_PBM` and `models_HTS` folders, respectively. We have saved the model weights in this repository if you want to skip training and proceed directly to prediction.

### Prediction

**Step 4.** To generate predictions, you need to run the script `make_predict.py` with the argument `--type_exp` set to 'PBM' or 'HTS', which specifies based on which experiments the prediction will be made.

| Prediction | Based on         | Discipline   |
|------------|------------------|--------------|
| PBM        | PBM              | Secondary    |
| GHTS       | PBM and HTS      | Primary      |
| CHS        | PBM and HTS      | Primary      |
| HTS        | HTS              | Secondary    |

As with the training mode, please run the second command after the first command has completed:

```
python make_predict.py --type_exp PBM
```
```
python make_predict.py --type_exp HTS
```

As a result of predictions on different models, the sum of the predictions is calculated and min-max scaling is applied. To merge the prediction results, run the script:

```
The command will be here.
```

The prediction results are saved on Google Drive.
