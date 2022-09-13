# Train a wgan-gp for silhouette generation
```
cd ./wgan-gp-pytorch
python train.py --train_dir [train_dir] --validation_dir [val_dir] --output_path [output_dir] --dim 64 --saving_step 300 --num_workers 8
```

# generate silhouette to attack on GaitSet and evaluate
preparation
```
cd ./silhouette-attack
python pretreatment.py --input_path=[root_path_of_raw_dataset] --output_path=[root_path_for_output]
```

for attack, edit the dataset_path in config.py to your 'root_path_for_output'
```
python generate.py
```

for test, edit the dataset_path in config.py to your 'output' contains generated images
to test on CASIA-A, edit config.py, 'dataset': 'CASIA-A', 'pid_num': 0,
```
python test_casiaB.py (python test_casiaA.py)
```

# black-box test on GaitGAN
```
cd ./gaitgan-pytorch/src
python gei.py
python transform.py
python knn_class_per_angle.py
```

# silhouette to video generation
```
cd ./spade
python train.py --name gait --dataset_mode custom --label_dir "/data1/" --image_dir "/data2/" --label_nc 1 --no_instance --contain_dontcare_label --use_vae --batchSize 1
python test.py --name gait --dataset_mode custom --label_dir "/gait/craft_resize/" --image_dir "/gait/craft_video/" --label_nc 1 --no_instance --contain_dontcare_label --use_vae --batchSize 16 --results_dir "/SPADE/results_craft/" --no_pairing_check
```
