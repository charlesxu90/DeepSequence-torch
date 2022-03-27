"""
Adapted from: https://github.com/debbiemarkslab/DeepSequence/blob/master/examples/run_svi.py
"""
from utils import DataHelper
from model import VariationalAutoencoder
from trainer import Trainer
import logging
import argparse
import os

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    data = DataHelper(alignment_file=args.alignment_file, calc_weights=True)
    vae_model = VariationalAutoencoder(data.seq_len, data.alphabet_size, latent_dim=30,
                                       enc_h1_dim=1500, enc_h2_dim=1500,
                                       dec_h1_dim=100, dec_h2_dim=500, dec_scale_mu=0.001)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    trainer = Trainer(vae_model, output_dir=args.out_dir, learning_rate=1e-3, betas=(0.9, 0.999), grad_norm_clip=1.0,
                      print_step_size=100)
    trainer.train(data, n_steps=300000)

def get_args():
    parser = argparse.ArgumentParser(description='Train a DeepSequence VAE model from alignment file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alignment_file', type=str, help='Full path to alignment file')
    parser.add_argument('-o', '--out_dir', type=str, help='Output dir path')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--n_steps', default=300000, type=int, help='Num of training steps, default=300000')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
