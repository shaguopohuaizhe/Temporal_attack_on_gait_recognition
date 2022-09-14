# Improved Training of Wasserstein GANs in Pytorch

This is forked from [Improved Training of Wasserstein GANs](https://github.com/steven-lang/food-interpolator/tree/cbe856c49df9dea072a3a399529d1848bc4bf0aa/improved-wgan-pytorch).

# Run

* Example:

**Fresh training**
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --train_dir /path/to/train --validation_dir /path/to/validation/ --output_path /path/to/output/ --dim 64 --saving_step 300 --num_workers 8
```

**Continued training:**
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --train_dir /path/to/train --validation_dir /path/to/validation/ --output_path /path/to/output/ --dim 64 --saving_step 300 --num_workers 8 --restore_mode --start_iter 5000
```
