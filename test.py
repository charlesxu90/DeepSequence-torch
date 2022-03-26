"""
Adapted from: https://github.com/debbiemarkslab/DeepSequence/blob/master/examples/run_svi.py
"""
import sys
import numpy as np
from utils import DataHelper
from model import VariationalAutoencoder
from trainer import Trainer
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # alignment_file = "./datasets/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m"
    alignment_file = "./datasets/BLAT_ECOLX_test.a2m"
    data = DataHelper(alignment_file=alignment_file, calc_weights=True)
    logger.info(f"Neff = {data.Neff}")

    vae_model = VariationalAutoencoder(data.seq_len, data.alphabet_size, latent_dim=30,
                                       enc_h1_dim=1500, enc_h2_dim=1500, dec_h1_dim=100, dec_h2_dim=500,
                                       dec_scale_mu=0.001)

    trainer = Trainer(vae_model, output_dir='./datasets', learning_rate=3e-4, betas=(0.9, 0.999), grad_norm_clip=1.0,
                      multi_gpu=True, print_step_size=1)
    trainer.train(data, n_steps=300000)
    # vae_model.save_parameters()


if __name__ == "__main__":
    main()
