# NovaBind
Salimov and Frolov Laboratory — participants in the international [Ibis](https://ibis.autosome.org) competition for predicting transcription factor binding levels to DNA sequences. We took part in predicting on genome sequences, based on artificial data.  More details about the architecture and methods used can be found in [google-document](https://clck.ru/3Ddv7i).

Here we demonstrate the performance of NovaBind as we predicted the data in the competition. Due to time constraints, you may notice some 'oddities', such as a different number of models in the ensemble for various TFs. This will be addressed in future versions of <b>NovaBind</b>.

## Environment
We used a server with a GPU running Ubuntu 20.04.6. To set up the environment, please use the following command:

> conda env create -f environment.yml

Don't forget to install the necessary GPU drivers in the environment. We have commented out the lines for driver installation based on our hardware.

## Input data
You can find the input data on the [Ibis site](https://ibis.autosome.org/download_data/final). The archive is too large, so we are not attaching it here. However, we have prepared test data to verify the correct execution of <b>NovaBind</b> training and prediction.
