#!/bin/sh
python test.py --name gait --dataset_mode custom --label_dir "/gait/craft_resize/" --image_dir "/gait/craft_video/" --label_nc 1 --no_instance --contain_dontcare_label --use_vae --batchSize 16 --results_dir "/SPADE/results_craft/" --no_pairing_check
