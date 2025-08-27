# MSF-TCMA

## Abstract

This is the source code for paper "_Temporal Downscaling Meteorological Variables to Unseen Moments: Continuous Temporal Downscaling via Multi-Source Spatial-Temporal-Wavelet Feature Fusion and Time-Continuous Manifold_", which is in minor revise stage of "_ISPRS Journal of Photogrammetry and Remote Sensing_"

## Installation

```
pip install -r requirements.txt
```

## Overview

- `model/model.py:` The algorithm architecture of this MSF-TCMA.
- `model/FrameExtraction.py:` The multiscale deep-wavelet feature extraction branch.
- `model/CrossFrameAttention.py:` The cross-modal spatiotemporal information fusion branch.
- `model/FlowEstimate.py:` The time-continuous manifold sampling branch.
- `config.py：`  Training configs for the MSF-TCMA.
- `train.py:` Train the MSF-TCMA.

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

## Acknowledgments

Our code style is based on [SGM-VFI]([https://github.com/chengtan9907/OpenSTL](https://github.com/MCG-NJU/SGM-VFI)). We sincerely appreciate for their contributions.

