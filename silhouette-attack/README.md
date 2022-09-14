# Silhouette attack

This repo is mainly based on attacking the sota model [GaitSet](https://github.com/AbnerHqC/GaitSet).

Preparation
```
cd ./silhouette-attack
python pretreatment.py --input_path=[root_path_of_raw_dataset] --output_path=[root_path_for_output]
```

To attack, edit the dataset_path in config.py to your 'root_path_for_output'
```
python generate.py
```

To test, edit the dataset_path in config.py to your 'output' contains generated images
to test on CASIA-A, edit config.py, 'dataset': 'CASIA-A', 'pid_num': 0,
```
python test_casiaB.py (python test_casiaA.py)
```
