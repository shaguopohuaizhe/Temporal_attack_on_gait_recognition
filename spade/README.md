# Video generation
This step uses [SPADE](https://github.com/NVlabs/SPADE) to translate silhouettes to videos.

```
cd ./spade
python train.py --name gait --dataset_mode custom --label_dir "/data1/" --image_dir "/data2/" --label_nc 1 --no_instance --contain_dontcare_label --use_vae --batchSize 1
python test.py --name gait --dataset_mode custom --label_dir "/gait/craft_resize/" --image_dir "/gait/craft_video/" --label_nc 1 --no_instance --contain_dontcare_label --use_vae --batchSize 16 --results_dir "/SPADE/results_craft/" --no_pairing_check
```
