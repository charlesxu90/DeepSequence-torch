from model import VariationalAutoencoder, load_model
from utils import DataHelper
from predictor import MutationEffectPredictor
import torch
import argparse

def main(args):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    data = DataHelper(alignment_file=args.alignment_file, calc_weights=False)

    vae_model = VariationalAutoencoder(data.seq_len, data.alphabet_size, latent_dim=30,
                                       enc_h1_dim=1500, enc_h2_dim=1500,
                                       dec_h1_dim=100, dec_h2_dim=500, dec_scale_mu=0.001)
    vae_model = load_model(vae_model, args.model_path, device, copy_to_cpu=True)
    predictor = MutationEffectPredictor(data, vae_model)

    n_iter = args.n_iter
    print(predictor.get_variant_delta_elbo(vae_model, [(32, "K", "F")], n_iter=n_iter))
    # -2.03463650668
    print(predictor.get_variant_delta_elbo(vae_model, [(32, "K", "F"), (33, "D", "N")], n_iter=n_iter))
    # -16.058655309


def get_args():
    parser = argparse.ArgumentParser(description='Predict mutation effect of variants with delta elbo',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alignment_file', type=str, help='Full path to alignment file')
    parser.add_argument('-m', '--model_path', type=str, help='Full path to VAE model checkpoint')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--n_iter', default=500, type=int, help='Num of iterations to calculate delta_elbo of variants,'
                                                                  'default=500')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
