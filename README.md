# MSF-TCMA

## Abstract

This is the source code for paper "_Temporal Downscaling Meteorological Variables to Unseen Moments: Continuous Temporal Downscaling via Multi-Source Spatial-Temporal-Wavelet Feature Fusion and Time-Continuous Manifold_", which is Published by "_ISPRS Journal of Photogrammetry and Remote Sensing_"

## Installation

```
pip install -r requirements.txt
```

## Overview

- `model/model.py:` The algorithm architecture of this MSF-TCMA.
- `model/FeatureExtractionBranch.py:` The multiscale deep-wavelet feature extraction branch.
- `model/InformationFusingBranch.py:` The cross-modal spatiotemporal information fusion branch.
- `model/ManifoldSamplingBranch.py:` The time-continuous manifold sampling branch.
- `makedataset.py:` Create the training/test dataset from the original dataset file.
- `config.py：`  Hyperparameters and training/test configs for the MSF-TCMA.
- `train.py:` Train the MSF-TCMA.
- `test.py:` Test the MSF-TCMA.

## Data preparation

The data used in this study and its processing have been described in detail in the manuscript, please see our paper for details.

## Train

After the data is ready, use the following commands to start training the model:
```
python train.py
```

## Test
Use the following commands to start testing the model:
```
python test.py
```

## Cite
If this project could help you, please cite:
@article{GAO202532,
title = {Temporal downscaling meteorological variables to unseen moments: Continuous temporal downscaling via Multi-source Spatial–temporal-wavelet feature Fusion and Time-Continuous Manifold},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {230},
pages = {32-54},
year = {2025},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2025.09.001},
url = {https://www.sciencedirect.com/science/article/pii/S092427162500351X},
author = {Sheng Gao and Lianlei Lin and Zongwei Zhang and Jiawei Wang},
}


## Acknowledgments

Our code style is based on [SGM-VFI]([https://github.com/chengtan9907/OpenSTL](https://github.com/MCG-NJU/SGM-VFI)). We sincerely appreciate for their contributions.
