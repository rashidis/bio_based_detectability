# Optimal visual search based on a model of target detectability in natural images

Optimal visual search based on a model of target detectability in natural images

This repository is the official implementation of [Optimal visual search based on a model of target detectability in natural images]. 

## Requirements

To install requirements:
```setup
pip install -r requirements.txt
```

## Detectability model
The presented model in the paper, outputs the detectability fall-off rate for any given image. 

To calculate the detectability of one single patch, run this command:
```produce the detectavility graphs
python get_ddash.py
```
The sample patches in the [datasets/test]() folder can be used (images taken from the texture dataset [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/). The model outputs the detectability-eccentricity graph of the input image and a .csv file with the image name and detectability fall-off rate. 

## Search model
To output the number of fixations and scanpath of any given textued image with the target pasted at an unknown location, run this command:
```produce the detectavility graphs
python visual_search.py
```
The sample csv files in the [datasets/test/simul_ddash_params.csv]() folder can be used as the input. The model outputs two csv files containing number of fixations and scanpth. 

## Results

Our model achieves the following performance on the 18 sample backgrounds:
| Model name         |       MSE       |         SE     |
| ------------------ |---------------- | -------------- |
| Alexnet + Log. Res.|      0.0978     |      0.0015    |





