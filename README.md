# Temporal attack on gait recognition

Implementation of our Pattern Recognition paper "Temporal Sparse Adversarial Attack on Sequence-based Gait Recognition".

### Table of Contents  
1) [Train a wgan-gp for silhouette generation](#Train-a-wgan-gp-for-silhouette-generation) <a name="Train-a-wgan-gp-for-silhouette-generation"/>
2) [Generate silhouette to attack on GaitSet and evaluate](#generate-silhouette-to-attack-on-GaitSet-and-evaluate) <a name="generate-silhouette-to-attack-on-GaitSet-and-evaluate"/>
3) [Black-box test on GaitGAN](#Black-box-test-on-GaitGAN) <a name="Black-box-test-on-GaitGAN"/>
4) [Silhouette to video generation](#Silhouette-to-video-generation)  <a name="Silhouette-to-video-generation"/>

## Train a wgan-gp for silhouette generation
```
cd ./wgan-gp-pytorch
python train.py --train_dir [train_dir] --validation_dir [val_dir] --output_path [output_dir] --dim 64 --saving_step 300 --num_workers 8
```

## Generate silhouette to attack on GaitSet and evaluate
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

## Black-box test on GaitGAN
```
cd ./gaitgan-pytorch/src
python gei.py
python transform.py
python knn_class_per_angle.py
```

## Silhouette to video generation
```
cd ./spade
python train.py --name gait --dataset_mode custom --label_dir "/data1/" --image_dir "/data2/" --label_nc 1 --no_instance --contain_dontcare_label --use_vae --batchSize 1
python test.py --name gait --dataset_mode custom --label_dir "/gait/craft_resize/" --image_dir "/gait/craft_video/" --label_nc 1 --no_instance --contain_dontcare_label --use_vae --batchSize 16 --results_dir "/SPADE/results_craft/" --no_pairing_check
```

## Citation
```
@article{HE2022109028,
title = {Temporal Sparse Adversarial Attack on Sequence-based Gait Recognition},
journal = {Pattern Recognition},
pages = {109028},
year = {2022},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2022.109028},
url = {https://www.sciencedirect.com/science/article/pii/S0031320322005088},
author = {Ziwen He and Wei Wang and Jing Dong and Tieniu Tan}
}
```
