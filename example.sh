#!/usr/bin/bash

python train.py. -a "./datasets/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m" \
  -o ./datasets

python predict.py -a "datasets/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m" \
  -m "datasets/model_149000_391.404.pt" --n_iter 100
