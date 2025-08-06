# Bayesian Structure Learning
This repository contains the code for generating the results as found in [this thesis](https://repository.tudelft.nl/record/uuid:83a071fd-8a33-464b-86d8-96186a2bbea6)

## Python files / Generating data / Display results
All the Python files are in the `scripts/` directory. These files do not contain any code related to the inferencing, only information about plotting and data creation.

To install the neccessary dependencies (preferrably in a virtual env), run
```shell
pip install -r requirements.txt
```

*create_data.py* - This file is used to generate the data. The relevant data is saved in a file with the name `global.txt`
*score_data.py* - This file can be used to score the inference results by comparing them to the results of YOLO having run over a full scene. The `pairs`-variable needs to be manually set using the first figure.
*performance_visualization.py* - Plots the results of `score_data.py`

All relevant paths are on the top of each file.

## Julia files / Performing inference
All Julia files are in the `src/` directory. This is where all the inferencing is done.

*structure_learning.jl* - This is the main file of interest. A must need reference is [this repo](https://github.com/biaslab/OnlineMessagePassingDirichletProcess).

## Work Flow
`create_data.py` -> `structure_learning.jl` -> `score_data.py` -> `performance_visualization.py`
