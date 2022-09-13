#!/bin/sh
python train.py --name gait --dataset_mode custom --label_dir "/data1/" --image_dir "/data2/" --label_nc 1 --no_instance --contain_dontcare_label --use_vae --batchSize 1
