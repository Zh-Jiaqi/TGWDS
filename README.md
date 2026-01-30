# TGWDS
TGWDS, a Terrain-Guided deep neural network for hub-height Wind field DownScaling. TGWDS incorporates a terrain-aware encoder to extract joint wind–terrain representations and a multi-stage decoder to integrate high- and low-frequency wind field structures. Model training is further enhanced by a spectral–physical hybrid loss that jointly balances wind magnitude, multi-frequency fidelity, and physical consistency. Experimental evaluations show that TGWDS achieves mean absolute errors of 0.05 m/s, 0.21 m/s, and 0.51 m/s for 2×, 4×, and 8× downscaling, respectively, with a maximal wind direction error of 5.9° under 8× downscaling. Notably, TGWDS demonstrates robust cross-scale generalization through conditioning on terrain, maintaining stable accuracy across unseen resolution configurations. Moreover, cross-regional transfer experiments spanning Southeast Asia and Western Europe show that TGWDS can be rapidly deployed to geographically and climatologically distinct areas without retraining, highlighting its strong adaptability and practical value. Efficiency tests further indicate that TGWDS completes 8× vector wind field downscaling for the entire Northeast China domain in 0.05 seconds on a single NVIDIA RTX 4090 GPU. Overall, TGWDS provides an accurate, efficient, and highly generalizable solution for producing high-resolution vector wind fields over large wind power regions, supporting real-time operational planning and short-term decision-making in modern wind energy systems.
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
