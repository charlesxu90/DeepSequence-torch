"""
Adapted from: https://github.com/debbiemarkslab/DeepSequence/blob/master/DeepSequence/model.py
VAE Reference: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from scipy.special import erfinv
from utils import get_path
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    def __init__(self, seq_len, alphabet_size, h1_dim, h2_dim, latent_dim):
        """ Encoder class in DeepSequence VAE"""
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.alphabet_size = alphabet_size
        logger.info(f"seq_len = {self.seq_len}, alphabet_size = {self.alphabet_size}")

        def init_W(input_dim, output_dim):
            return nn.Parameter(torch.empty((input_dim, output_dim), dtype=torch.float32)
                                .normal_(mean=0., std=np.sqrt(2.0/(input_dim+output_dim))))

        def init_b(output_dim):
            return nn.Parameter(0.1 * torch.ones(output_dim, dtype=torch.float32))

        def init_W_log_sigma(input_dim, output_dim):
            return nn.Parameter(-5 * torch.ones(input_dim, output_dim, dtype=torch.float32))

        def init_b_log_sigma(output_dim):
            return nn.Parameter(-5 * torch.ones(output_dim, dtype=torch.float32))

        self.W1 = init_W(self.seq_len*self.alphabet_size, h1_dim)
        self.b1 = init_b(h1_dim)
        self.h1_relu = nn.ReLU()

        self.W2 = init_W(h1_dim, h2_dim)
        self.b2 = init_b(h2_dim)
        self.h2_relu = nn.ReLU()

        self.W_z_mu = init_W(h2_dim, latent_dim)
        self.b_z_mu = init_b(latent_dim)

        self.W_z_log_sigma = init_W(h2_dim, latent_dim)
        self.b_z_log_sigma = init_b_log_sigma(latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.reshape((batch_size, self.seq_len * self.alphabet_size))
        h1 = self.h1_relu(x_reshaped.matmul(self.W1) + self.b1)
        h2 = self.h2_relu(h1.matmul(self.W2) + self.b2)

        z_mu = h2.matmul(self.W_z_mu) + self.b_z_mu
        z_log_sigma = h2.matmul(self.W_z_log_sigma) + self.b_z_log_sigma
        # logger.info(f"z_log_sigma {z_log_sigma.mean().cpu().detach().numpy()}")

        return z_mu, z_log_sigma


class SparseDecoder(nn.Module):
    def __init__(self, seq_len, alphabet_size, latent_dim, h1_dim=100, h2_dim=500, n_tiles=4, conv_size=40,
                 scale_mu=0.01, scale_sigma=4.0):
        """
            Sparse Decoder in DeepSequence VAE
        Parameters
        ----------
        seq_len         Sequence length
        alphabet_size   Alphabet size
        latent_dim      Latent dimension, i.e. length of z
        h1_dim          First hidden layer dimension
        h2_dim          Second hidden layer dimension
        n_tiles         Number of tiles for sparsity calculation in the output layer
        conv_size       Convolution size, for C parameter estimation, i.e. global AA-AA relation estimation
        scale_mu        Mean for lamda, the scale scalar in the output layer
        scale_sigma     Variance for lambda, the Scale scalar in the output layer
        """
        super(SparseDecoder, self).__init__()
        self.alphabet_size = alphabet_size
        self.seq_len = seq_len
        self.h2_dim = h2_dim * n_tiles  # k in Sij calculation where `j mod (H/k)`. default=4
        self.n_tiles = n_tiles
        self.conv_size = conv_size

        def init_W(input_dim, output_dim):
            return nn.Parameter(torch.empty((input_dim, output_dim), dtype=torch.float32) \
                                .normal_(mean=0., std=np.sqrt(2.0/(input_dim+output_dim))))

        def init_b(output_dim):
            return nn.Parameter(0.1 * torch.ones(output_dim, dtype=torch.float32))

        def init_W_log_sigma(input_dim, output_dim):
            return nn.Parameter(-5 * torch.ones(input_dim, output_dim, dtype=torch.float32))

        def init_b_log_sigma(output_dim):
            return nn.Parameter(-5 * torch.ones(output_dim, dtype=torch.float32))

        def init_W_out_scale_mu(input_dim, output_dim):  # init with zeros
            return nn.Parameter(torch.ones(input_dim, output_dim, dtype=torch.float32))

        # First sparse layer
        self.W1_mu = init_W(latent_dim, h1_dim)
        self.b1_mu = init_b(h1_dim)
        self.W1_log_sigma = init_W_log_sigma(latent_dim, h1_dim)
        self.b1_log_sigma = init_b_log_sigma(h1_dim)
        self.h1_relu = nn.ReLU()

        # Second sparse layer
        self.W2_mu = init_W(h1_dim, self.h2_dim)
        self.b2_mu = init_b(self.h2_dim)
        self.W2_log_sigma = init_W_log_sigma(h1_dim, self.h2_dim)
        self.b2_log_sigma = init_b_log_sigma(self.h2_dim)
        self.h2_sigmoid = nn.Sigmoid()

        # Output layer, two operations to update weight: conv and sparse
        self.W_out_mu = init_W(self.h2_dim, self.seq_len*self.conv_size)
        self.W_out_log_sigma = init_W_log_sigma(self.h2_dim, self.seq_len*self.conv_size)

        # Convolution for the weights in output layer
        self.W_conv_mu = init_W(self.conv_size, self.alphabet_size)
        self.W_conv_log_sigma = init_W_log_sigma(self.conv_size, self.alphabet_size)

        # Sparsity parameters, i.e. fadeout with tiles, for the weights in the output layer
        self.W_out_scale_mu = init_W_out_scale_mu(int(self.h2_dim/n_tiles), self.seq_len)
        self.W_out_scale_log_sigma = init_W_log_sigma(int(self.h2_dim/n_tiles), self.seq_len)
        self.out_scale_sigmoid = nn.Sigmoid()

        self.b_out_mu = init_b(self.seq_len*self.alphabet_size)
        self.b_out_log_sigma = init_b_log_sigma(self.seq_len*self.alphabet_size)

        # Scale priors, for KL div calculation
        self.scale_prior_mu = np.sqrt(2.0) * scale_sigma * erfinv(2.0 * scale_mu - 1.0)
        self.scale_prior_log_sigma = np.log(scale_sigma)

        self.final_pwm_scale_mu = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.final_pwm_scale_log_sigma = nn.Parameter(-5 * torch.ones(1, dtype=torch.float32))
        self.final_softmax = nn.Softmax(dim=-1)
        self.final_log_softmax = nn.LogSoftmax(dim=-1)

    @staticmethod
    def _sampler(mu, log_sigma):
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sigma) * eps

    @staticmethod
    def _KLD_diag_gaussians(mu, log_sigma, prior_mu=0., prior_log_sigma=0.) -> Tensor:
        """ KL divergence between two Diagonal Gaussians """
        return prior_log_sigma - log_sigma + \
               0.5 * (torch.exp(2.*log_sigma) + torch.square(mu - prior_mu)) * torch.exp(-2.*prior_log_sigma*torch.ones(1)) - 0.5

    def _get_KLD_from_param(self, mu, log_sigma) -> Tensor:
        """ KL divergence between two Diagonal Gaussians """
        return torch.sum(-self._KLD_diag_gaussians(mu.flatten(), log_sigma.flatten(), 0., 0.))

    def get_KL_div(self):
        KL_div = self._get_KLD_from_param(self.W1_mu, self.W1_log_sigma)
        KL_div += self._get_KLD_from_param(self.b1_mu, self.b1_log_sigma)
        KL_div += self._get_KLD_from_param(self.W2_mu, self.W2_log_sigma)
        KL_div += self._get_KLD_from_param(self.b2_mu, self.b2_log_sigma)

        KL_div += self._get_KLD_from_param(self.W_conv_mu, self.W_conv_log_sigma)
        KL_div += self._get_KLD_from_param(self.W_out_mu, self.W_out_log_sigma)
        KL_div += self._get_KLD_from_param(self.b_out_mu, self.b_out_log_sigma)
        KL_div += self._get_KLD_from_param(self.final_pwm_scale_mu, self.final_pwm_scale_log_sigma)

        # Use a continuous relaxation of a spike and slab prior with a logit normit scale distribution
        KL_div_scale = -self._KLD_diag_gaussians(self.W_out_scale_mu, self.W_out_scale_log_sigma,
                                                 self.scale_prior_mu, self.scale_prior_log_sigma)
        KL_div += torch.sum(KL_div_scale)

        return KL_div

    def forward(self, z):
        W1 = self._sampler(self.W1_mu, self.W1_log_sigma)
        b1 = self._sampler(self.b1_mu, self.b1_log_sigma)
        z_h1 = self.h1_relu(z.matmul(W1) + b1)

        W2 = self._sampler(self.W2_mu, self.W2_log_sigma)
        b2 = self._sampler(self.b2_mu, self.b2_log_sigma)
        z_h2 = self.h2_sigmoid(z_h1.matmul(W2) + b2)

        # Process the weights for the output
        W_out = self._sampler(self.W_out_mu, self.W_out_log_sigma)

        # Sample the convolution weights, & make the matrix the size of the output
        W_conv = self._sampler(self.W_conv_mu, self.W_conv_log_sigma)  # C
        W_out = W_out.reshape(self.h2_dim*self.seq_len, self.conv_size).matmul(W_conv)

        # Apply group sparsity `S_i \odot h^{(2)}`
        W_scale = self._sampler(self.W_out_scale_mu, self.W_out_scale_log_sigma)  # \tilde S_{ij}
        W_scale = torch.tile(W_scale, (self.n_tiles, 1))   # \tilde S_{j  mod H/k,i}
        W_scale = self.out_scale_sigmoid(W_scale.reshape(W_scale.shape[0],W_scale.shape[1], 1))  # S_{ij}

        W_out = W_out.reshape(self.h2_dim, self.seq_len, self.alphabet_size) * W_scale
        W_out = W_out.reshape(self.h2_dim, self.seq_len * self.alphabet_size)

        b_out = self._sampler(self.b_out_mu, self.b_out_log_sigma)
        x_reconstructed_flat = z_h2.matmul(W_out) + b_out.reshape(1, b_out.shape[0])  # be careful of b_out shape

        pwm_scale = self._sampler(self.final_pwm_scale_mu, self.final_pwm_scale_log_sigma)  # tilde lambda
        x_reconstructed_flat = x_reconstructed_flat * torch.log(1.0 + torch.exp(pwm_scale))

        x_reconstructed_unorm = x_reconstructed_flat.reshape(z_h2.shape[0], self.seq_len, self.alphabet_size)
        x_reconstructed_norm = x_reconstructed_unorm - torch.max(x_reconstructed_unorm, dim=2, keepdim=True).values

        x_reconstructed = self.final_softmax(x_reconstructed_norm)
        log_logits = self.final_log_softmax(x_reconstructed_norm)

        return x_reconstructed, log_logits


class VariationalAutoencoder(nn.Module):
    def __init__(self, seq_len, alphabet_size, latent_dim=30, enc_h1_dim=1500, enc_h2_dim=1500,
                 dec_h1_dim=100, dec_h2_dim=500, dec_scale_mu=0.001):
        """ DeepSequence VAE class. """
        super(VariationalAutoencoder, self).__init__()

        self.encoder = Encoder(seq_len, alphabet_size, enc_h1_dim, enc_h2_dim, latent_dim)
        self.decoder = SparseDecoder(seq_len, alphabet_size, latent_dim, dec_h1_dim, dec_h2_dim,
                                     n_tiles=4, conv_size=40, scale_mu=dec_scale_mu, scale_sigma=4.0)

    @staticmethod
    def _sampler(mu, log_sigma):
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sigma) * eps

    def forward(self, x):
        z_mu, z_log_sigma = self.encoder(x)
        z = self._sampler(z_mu, z_log_sigma)
        # logger.info(f"z_mu {z_mu.mean().cpu().detach().numpy()}, z_log_sigma {z_log_sigma.mean().cpu().detach().numpy()}")
        z_KL_div = 0.5 * torch.sum(1.0 + 2.0*z_log_sigma - z_mu**2.0 - torch.exp(2.0*z_log_sigma), dim=1)

        x_reconstructed, x_log_logits = self.decoder(z)
        logp_xz = torch.sum(torch.sum(x * x_log_logits, dim=-1), dim=-1)
        x_KL_div = self.decoder.get_KL_div()

        return x_reconstructed, logp_xz, z_KL_div, x_KL_div

    def configure_optimizers(self, learning_rate=1e-3, betas=(0.9, 0.999)):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=betas)
        # optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=betas)
        return optimizer

    def get_likelihoods(self, x):
        z_mu, z_log_sigma = self.encoder(x)
        z = self._sampler(z_mu, z_log_sigma)
        z_KL_div = 0.5 * torch.sum(1.0 + 2.0*z_log_sigma - z_mu**2.0 - torch.exp(2.0*z_log_sigma), dim=1)

        _, x_log_logits = self.decoder(z)
        logp_xz = torch.sum(torch.sum(x * x_log_logits, dim=-1), dim=-1)
        logp_x = logp_xz + z_KL_div

        return logp_x


def save_model(model, base_dir, base_name):
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), get_path(base_dir, base_name, '.pt'))


def load_model(model, model_weights_path, device, copy_to_cpu=True):
    raw_model = model.module if hasattr(model, "module") else model
    map_location = lambda storage, loc: storage if copy_to_cpu else None
    raw_model.load_state_dict(torch.load(model_weights_path, map_location))
    return raw_model.to(device)