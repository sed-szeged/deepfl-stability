# Supplemental Material for "On the Stability and Applicability of Deep Learning in Fault Localization"

This repository contains open science data which ensures full reproducibility of the research. The paper was submitted at the [31th International Conference on Software Analysis, Evolution and Reengineering (SANER)](https://conf.researchr.org/home/saner-2024). If you use the data or models for academic purposes, please cite the appropriate publication:
```
@inproceedings{csuvik:deepfl,
 title = {On the Stability and Applicability of Deep Learning in Fault Localization},
 author = {Csuvik, Viktor and Roland, Aszmann and Árpád, Beszédes and Ferenc, Horváth and Tibor, Gyimóthy},
 booktitle={2024 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER)}, 
 year = {2024},
 doi = {},
}
```

## Source code

The [code](code) directory contains the source code for calculating the ranks for statements, it is code from https://github.com/ICSE2022FL/ICSE2022FLCode with adding:
- Ability for using fixed random seeds.
- The simplified models for neural networks. 
- The [seeds](code/seeds) directory contains every used random seeds, in format: seed-{program}-{version}-{train}.txt, for example: "seed-Chart-1-5.txt", which means version 1 of Chart, 5. run of a model.

To run the code:
There is a need to download the dataset from https://fault-localization.cs.washington.edu/data/. We included a [script](code/download_d4j_data.py) to help, it downloads the dataset and extract all needed files to exact location.

To run a model for fault-localization:

```run.py -d {dataset} -p {program} -i {version} -m {model} -e {experiment} -r {random_seed}```

| name |   meaning   |                             value                             |
|:----:|:-----------:|:-------------------------------------------------------------:|
|  -d  |   dataset   |                             "d4j"                             |
|  -p  |   program   |            "Chart", "Lang", "Math", "Mockito", "Time",        |
|  -i  |   version   |                         "1", "2", ...                         |
|  -m  |    model    | "MLP-FL", "MLP-FLS", "CNN-FL", "CNN-FLS", "RNN-FL", "RNN-FLS" |
|  -e  | experiment  |                    "origin", "resampling"                     |
|  -r  | random seed |                  see in [seeds](code/seeds)                   |

{model} with "S" endings means the simplified version of the neural network model. 

Example:

```run.py -d d4j -p Chart -i 10 -m MLP-FL -e origin -r 3494891057880294974```

## Results

The [results](results) directory contains detailed results of our experiments.
- In [figures](results/figures) directory:
  - [boxplots](results/figures/boxplots) contains the detailed box plots for every project mentioned in Section VI.A, and for all improvements in section VII. as well,
  - [churn-histograms](results/figures/churn-histograms) contains histograms for every examined churn setting, related to Section VI.B and VII. (Figure 4. and 5.).
- In [raw](results/raw) are the raw results, in csv format, in various subdirectories:
  - [churn](results/raw/churn) contains all examined variation of churn, for every model and version in table format,
  - [means](results/raw/means) contains the mean values of DLFL, related to Section VI.A and VII.(Table III., VI.),
  - [statistics](results/raw/statistics) contains the significance testing statistics comparing minimal with maximal expense values, mentioned at Section VI.A and VII.,
  - [top](results/raw/top) contains the *Top-N* values, related to Section VI.A and VII.(Table IV, VII),
  - [variety](results/raw/variety) contains the variety of same expense appearing results for Section VI.A and VII.(Table II., V.),
  - The [baseline](results/raw/origin_baseline.csv), [simplified](results/raw/origin_simplified.csv), [resampling](results/raw/resampling_basic.csv), [resampling-simplified](results/raw/resampling_simplified.csv) files contains the raw expense results of each run.
 
[![DOI](https://zenodo.org/badge/719515916.svg)](https://zenodo.org/doi/10.5281/zenodo.10496188)
