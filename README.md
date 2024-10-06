# NovaBind
Salimov and Frolov Laboratory — participants in the international [Ibis](https://ibis.autosome.org) competition for predicting transcription factor binding levels to DNA sequences. We took part in predicting on genome sequences, based on artificial data.  More details about the architecture and methods used can be found in [google-document](https://clck.ru/3Ddv7i).

Here we demonstrate the performance of NovaBind as we predicted the data in the competition. Due to time constraints, you may notice some 'oddities', such as a different number of models in the ensemble for various TFs. This will be addressed in future versions of <b>NovaBind</b>.

## Environment
We used a server with a GPU running Ubuntu 20.04.6. To set up the environment, please use the following command:

```
conda env create -f environment.yml
```

## Input data
You can find the input data on the [Ibis site](https://ibis.autosome.org/download_data/final). The archive is too large, so we are not attaching it here. Please download the archive, unzip it, and place the `data` folder into the root folder where all the scripts from the repository are located.

## Training repeoduction

### Data preprocessing
*Step 1.* It is necessary to extract the files and convert them to a unified .csv format. For future model ensembling, we will immediately split the data into folds. All unnecessary and temporary files are deleted, leaving only the `folds_PBM` and `folds_HTS` directories with the necessary data. 

To run the script that does this, execute the following command in bash:

```
python prep_data.py
```

