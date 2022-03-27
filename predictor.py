from utils import DataHelper
import numpy as np
import torch
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MutationEffectPredictor:
    def __init__(self, data: DataHelper, model):
        self.data = data
        self.wt_elbo = None  # Use to calculate delta_elbo: mutant_elbo - wt_elbo

        # The WT sequence name is expected to be like: >[NAME]/[start]-[end], corresponding to Uniprot start and end
        focus_loc = self.data.wt_seq_name.split("/")[-1]
        focus_start, _ = (int(pos) for pos in focus_loc.split("-"))

        # Map focus columns to wild-type amino acids
        self.focus_col2aa = {idx_col+focus_start: data.wt_seq[idx_col] for idx_col in data.focus_cols}
        self.focus_col_list = list(self.focus_col2aa.keys())
        # Map focus columns to index of focus columns in the focus region
        self.focus_col2focus_idx = {idx_col+focus_start: idx_col for idx_col in data.focus_cols}

        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        logger.info(f"device = {self.device}")
        self.model = torch.nn.DataParallel(model).to(self.device)

    def get_variant_seq(self, mutant_tuple_list):
        """Obtain the sequence of the variant from its mutant list"""
        mut_seq = self.data.focus_seq_trimmed
        for pos, wt_aa, mut_aa in mutant_tuple_list:
            mut_seq[self.focus_col2focus_idx[pos]] = mut_aa
        return mut_seq

    def get_variant_delta_elbo(self, model, variant_mutations, n_iter=10):
        """
            Predict mutation effect of a single variant with the list of mutations
        :param model:               A DeepSequence VAE model, type of VariationalAutoencoder
        :param variant_mutations:   A list of all mutations in a variant, e.g. [(126, "G", "A"), (137, "I", "P")]
        :param n_iter:              Number of prediction iterations, will take mean of them, default=10
        :return:                    Delta elbo of the variant, i.e. variant_elbo - wt_elbo
        """
        for pos, wt_aa, mut_aa in variant_mutations:
            if pos not in self.focus_col_list or self.focus_col2aa[pos] != wt_aa:
                print("Not a valid mutant!", pos, wt_aa, mut_aa)
                return None

        mut_seq = self.get_variant_seq(variant_mutations)
        mutant_seqs = [self.data.focus_seq_trimmed, mut_seq] if self.wt_elbo is None else [mut_seq]
        mutant_one_hot = self.data.seqs2onehot(mutant_seqs)

        x_mutant_one_hot = torch.Tensor(mutant_one_hot).to(self.device)
        prediction_matrix = np.zeros((x_mutant_one_hot.shape[0], n_iter))
        model.eval()
        for i in range(n_iter):
            with torch.no_grad():
                batch_preds = model.get_likelihoods(x_mutant_one_hot)
                prediction_matrix[:, i] = batch_preds.cpu().detach().numpy()

        mean_elbos = np.mean(prediction_matrix, axis=1).flatten().tolist()  # Take the mean of all iterations
        self.wt_elbo = mean_elbos.pop(0) if self.wt_elbo is None else self.wt_elbo

        return mean_elbos[0] - self.wt_elbo
