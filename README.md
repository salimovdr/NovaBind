# NovaBind
The Salimov and Frolov Laboratory — winners in the international [Ibis](https://ibis.autosome.org) competition for predicting transcription factor binding levels to DNA sequences. We participated in predictions on genome sequences, utilizing synthetic data. More details about the architecture and methods used can be found in this [Google Document](https://clck.ru/3Ddv7i).

Here we demonstrate the performance of **NovaBind** as we predicted the data in the competition.

## Environment
We used a server with a GPU running Ubuntu 20.04.6. To set up the environment, please use the following command:

```bash
conda env create -f environment.yml
```

## Input data
You can find the input data on the [Ibis site](https://ibis.autosome.org/download_data/final). The archives is too large, so we are not attaching it here. Please download it, unzip it, and place the `data` folder in the root directory where all the repository scripts are located.

## Reproduction

### Data preprocessing
**Step 1.** It is necessary to extract the files and convert them to a unified .csv format. For future model ensembling, we immediately split the data into folds. The `folds_PBM`, `folds_HTS` and `test` directories with the necessary data are created. 

To run the script that does this, execute the following command in bash:

```bash
python prep_data.py
```

**Step 2.** We are ready to split the folds into training and validation sets. DNA sequences are encoded using one-hot encoding. The complementary sequences are added to the data. For the data from the GHTS and CHS experiments, sequence segmentation is performed using a sliding window with stride of 1. These actions are performed in the `encode_data.py` script:

```bash
python encode_data.py
```

### Training
NA for SNP prediction.

### Prediction

**Step 4.** To generate predictions, you need to run the script `make_predict.py` with the argument `--type_exp` set to 'PBM' or 'HTS', which specifies based on which experiments the prediction will be made.

| Prediction | Based on         | Discipline   |
|------------|------------------|--------------|
| PBM        | PBM              | Secondary    |
| GHTS       | PBM and HTS      | Primary      |
| CHS        | PBM and HTS      | Primary      |
| HTS        | HTS              | Secondary    |

If you want to run predictions based on PBM or HTS in parallel, please specify the device number to perform the calculations on:

```bash
python make_predict.py --device 0 --type_exp PBM
python make_predict.py --device 1 --type_exp HTS
```

As a result of predictions on different models, the sum of the predictions is calculated and min-max scaling is applied. To merge the prediction results, run the script:

```bash
python get_results.py
```
