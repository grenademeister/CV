# UNet-Based Segmentation

This repository implements an end-to-end semantic segmentation pipeline using the U-Net architecture on the Oxford Pet dataset.

## Requirements

- Python 3.11+
- PyTorch
- torchvision
- pyyaml
- numpy
- matplotlib
- tqdm

## Project Structure

```
config.yaml        # Training and model configuration
dataset.py         # PyTorch Dataset for Oxford Pet & trimaps
unet.py            # U-Net model definition
train.py           # Training loop with logging, scheduler, early stopping
test.py            # Inference & quantitative evaluation
seg.py             # Visualization script for qualitative results
var/               # Generated logs & checkpoints
data_pet/          # Oxford Pet images and annotations
```
