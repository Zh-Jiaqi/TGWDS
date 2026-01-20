# TGWDS
A machine learning method for wind downscaling

*2025.12*

## Abstract

![Graphabstract](data/intro.jpg)



## Installation

```
conda create -n wpn python=3.8
conda activate tgwds
git clone https://github.com/Zhang-zongwei/TGWDS.git
cd TGWDS
pip install -r requirements.txt
```

## Overview

- `data/:` contains a test set for the northeast region of the manuscript, which can be downloaded via the link .
- `models:` contains the network architecture of this TGWDS.
- `utils/:` contains data processing files and some layers module.
- `exp/:` contains weights, predicted wind speed results and evaluation methods.
- `config.py：`  training configs for the TGWDS.
- `train.py:` Train the TGWDS.
- `test.py` Test the TGWDS.

## Data preparation
The data used in this study and its processing have been described in detail in the manuscript. To facilitate the testing, we have prepared the 
[dataset](https://drive.google.com/drive/folders/1SGTdGOfXfyum6rmT2DiQgd9UYXdTYeKc?usp=sharing).

## Train
After the data is ready, use the following commands to start training the model:
```
python train.py
```

## Test
After training is complete, run the following command to test:
```
python test.py
```
The test results will be generated in the `exp/results` folder. You can view the evaluation results by running the following command:
```
python wind_eval.py
```
## Acknowledgments

Our code is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL),[Restormer](https://github.com/swz30/Restormer). We sincerely appreciate for their contributions.
