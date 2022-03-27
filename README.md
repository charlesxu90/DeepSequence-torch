# PyTorch version of DeepSequence

## Installation
### Create environment
```commandline
conda env create -f environment.yml 
```
## Usage
Activate environment
```commandline
conda activate deepseq-env
```
Train DeepSequence VAE model from alignment
```commandline
python train.py. -a "./datasets/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m" \
  -o ./datasets
```
Predict with a DeepSequence VAE model
```commandline
python predict.py -a "datasets/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m" \
  -m "datasets/model_149000_391.404.pt" --n_iter 500
```

Any contributions are welcome!

Note: DeepSequence is not suitable for online use, as it requires n_iter (usually 500) of predictions to get an accurate delta_elbo.

## Special thanks
- Original repository: [DeepSequence](https://github.com/debbiemarkslab/DeepSequence) implemented with Theano.

License: [MIT License](LICENSE) 