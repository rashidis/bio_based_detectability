# Optimal visual search based on a model of target detectability in natural images

This repository is the code used to generate the results in the paper
[Optimal visual search based on a model of target detectability in natural images]. 
which was presented at [NeurIPS 2020](http://https://nips.cc/)

<img src="https://github.com/rashidis/bio_based_detectability/blob/main/arc.png" width="930">

## Requirements

To install requirements:
```setup
pip3 install -r requirements.txt
```

## Detectability model
The presented model in the paper outputs detectability as a function of eccentricity for any given image. 

To see how to calculate the detectability of one object on a set of backgrounds, run:

```produce the detectavility graphs
python get_ddash.py -h
```

This will use the object file given as the object_file parameter in the data/overlays fodler, and the backgrounds in the data/test folder.

Sample background patches can be found in [datasets/test](https://github.com/rashidis/bio_based_detectability/tree/main/data/test) (images taken from the texture dataset [ETHZ Synthesizability Dataset](http://people.ee.ethz.ch/~daid/synthesizability/#Downloads). The model outputs the detectability-eccentricity graph of the input image and a .csv file with the image name and detectability fall-off rate. 

## Search model
To see how to output the number of fixations and scanpath of any given textued image with the target pasted at an unknown location, run this command:

```produce the detectavility graphs
python visual_search.py -h
```

The sample input csv files provided (default parameters) in the files folder are from [datasets/test/simul_ddash_params.csv](https://github.com/rashidis/bio_based_detectability/blob/main/files/simul_ddash_params.csv). The model outputs two csv files containing number of fixations and scanpth. 

## Results

Our model achieves the following performance on the 18 sample backgrounds in figure 3 of the main paper:
| Model name         |       MSE       |         SE     |
| ------------------ |---------------- | -------------- |
| Alexnet + Log. Res.|      0.0978     |      0.0015    |





