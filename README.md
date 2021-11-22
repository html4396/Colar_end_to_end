## I2Sim: Modeling Intra-Segment and Intra-Category Similarity forPractical Online Action Detection

This repository is the official implementation of I2Sim. `In this work, we study the temporal action localization under single-background-frame supervision.` 

![Illustrating the architecture of the proposed I2Sim](framework.jpg)


## Requirements

a. To install requirements:

```setup
conda env create -n env_name -f environment.yaml
```

Before running the code, please activate this conda environment.

## Data Preparation

a. Enter data directory

	cd ${Colar_root}/data/thumos

b. Download raw annotations and video data

~~~~
${Colar_root}/data/thumos/download.sh
~~~~

c. Extract frames

```
cd ${Colar_root}/data/thumos14
${Colar_root}/data/extract_frames.sh videos/val frames/val -vf fps=25 -s 224x224 %05d.png
${Colar_root}/data/extract_frames.sh videos/test frames/test -vf fps=25 -s 224x224 %05d.png
```

## Train

a. Config

Modify 'anno path', 'frames path', 'visible cuda' and so on accordingly in the config file like

`./init.py`

b. Train

```train
python main.py --cuda_id 0
```
## Test

a. Config

Modify 'weight path' accordingly in the config file like

`./init.py`

b. Test

```eval
python test.py --cuda_id 0 --model '*.pth' 
```
## Pre-trained Models

You can download pretrained models here:

- [THUMOS14_best.pth](https://pan.baidu.com/s/1k8P2lUWLN3t6r2JUSSZb9Q) trained on THUMOS14  (pwd: yd9m)

