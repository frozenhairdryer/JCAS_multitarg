# JCAS_multitarg
Corresponding code to WSA/SCC 2023 contribution

## Init

Before you start, the figure folder has to be created. Run:

```
cd Autoencoder
mkdir figures
```
If you are planning on running the code on cpu, run:
```
pip install -r requirements.txt
```

If you want to run on gpu, cupy is installed additionally. Run:
```
pip install -r requirements_gpu.txt
```
Additionally, install pytorch according to their website.

## Overview

Training and plotting can be done by running split_test.py. We recommend running this file to get accustomed to the code.

Main Components are:

  * autoencoder_compare_cpr: evaluation script for sweeping number of samples

  * Folder Autoencoder:

    * training_routine has training function which is described below

    * NN_classes has the following functions and classes:
      * Encoder
      * Decoder (Comm Receiver)
      * custom binary loss function (equivalent to BCE loss)
      * Beamformer
      * Presence_Detector
      * Angle_Estimator


