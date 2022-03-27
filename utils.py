"""
Adapted from: https://github.com/debbiemarkslab/DeepSequence/blob/master/DeepSequence/helper.py
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_path(base_dir, base_name, suffix):
    return os.path.join(base_dir, base_name + suffix)

class DataHelper:
    def __init__(self, alignment_file="", wt_seq_name="", calc_weights=True, working_dir=".", theta=0.2) -> object:

        """
        Class to load and organize alignment data. This function also helps makes predictions about mutations.

        Params:
        --------------
        alignment_file:     Path to the alignment file
        wt_seq_name:        Name of the WT sequence in the alignment. Default: the first sequence in the alignment
        calc_weights:       Calculate sequence weights or not. Default: True
        working_dir:        Directory to save "params", "logs", and "embeddings"
        theta:              Sequence weighting hyperparameter. Default=0.2
                            Generally: 0.2 for prokaryotic and eukaryotic, and 0.01 for Viruses.
        """
        np.random.seed(42)
        self.alignment_file = alignment_file
        self.wt_seq_name = wt_seq_name
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.theta = theta  # Alignment processing parameters

        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"  # only protein supported
        self.alphabet_size = len(self.alphabet)

        self.aa_dict = {aa: i for i, aa in enumerate(self.alphabet)}  # AA to index
        self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}  # index to AA

        self.seq_name_to_sequence, self.wt_seq, self.focus_cols = self.gen_basic_alignment()
        self.seq_len = len(self.focus_cols)
        self.focus_seq_trimmed = [self.wt_seq[ix] for ix in self.focus_cols]

        self.x_train, self.x_train_names, self.weights = self.gen_full_alignment()
        self.Neff = np.sum(self.weights)
        logger.info(f"neff = {self.Neff}, Data Shape = {self.x_train.shape}")

    def read_alignment_file(self):
        """ Read the alignment file into a dict: {name -> seq}"""
        seq_name_to_sequence = {}

        name = ""
        INPUT = open(self.alignment_file, "r")
        for i, line in enumerate(INPUT):
            line = line.rstrip()
            if line.startswith(">"):
                name = line
            else:
                if name in seq_name_to_sequence.keys():
                    seq_name_to_sequence[name] += line
                else:
                    seq_name_to_sequence[name] = line
        INPUT.close()

        return seq_name_to_sequence

    def gen_basic_alignment(self):
        """ Read alignment file and obtain the focused columns, without trim """
        seq_name_to_sequence = self.read_alignment_file()
        logger.info(f"Totally {len(seq_name_to_sequence)} in the alignment.")

        if self.wt_seq_name == "":  # pick the first seq if no focus sequence
            self.wt_seq_name = list(seq_name_to_sequence.keys())[0]

        # Select focus columns These columns are the uppercase residues of the .a2m file
        wt_seq = seq_name_to_sequence[self.wt_seq_name]
        focus_cols = [ix for ix, s in enumerate(wt_seq) if s == s.upper()]

        return seq_name_to_sequence, wt_seq, focus_cols

    def remove_invalid_seqs(self):
        """ Remove invalid sequences in self.seq_name_to_sequence"""
        alphabet_set = set(list(self.alphabet))  # Remove sequences that have bad characters
        seq_names_to_remove = []
        for seq_name, sequence in self.seq_name_to_sequence.items():
            for letter in sequence:
                if letter not in alphabet_set and letter != "-":
                    seq_names_to_remove.append(seq_name)

        for seq_name in set(seq_names_to_remove):  # remove invalid seqs
            del self.seq_name_to_sequence[seq_name]

        logger.info(f"{len(self.seq_name_to_sequence)} sequences remains after removing invalid!")

    def seqs2onehot(self, seqs: list):
        """
         Encode the sequences into one_hot encoding
        :param      seqs:       List of sequences in the focus region
        :return:    x_enc:      One_hot encoding of the sequence
        """
        x_enc = np.zeros((len(seqs), len(self.focus_cols), len(self.alphabet)))
        for i, seq in enumerate(seqs):
            for j, letter in enumerate(seq):
                if letter in self.aa_dict:
                    k = self.aa_dict[letter]
                    x_enc[i, j, k] = 1.0
        return x_enc

    def gen_full_alignment(self):
        """ Obtain the focus seqs, remove invalid, and calculate weight"""
        # Get only the focus columns, built the model only on this region
        for seq_name, sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".", "-")  # Replace periods with dashes (the uppercase equivalent)
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]  # only use focus cols

        self.remove_invalid_seqs()  # remove invalid seqs in self.seq_name_to_sequence
        x_train_names = list(self.seq_name_to_sequence.keys())
        x_train = self.seqs2onehot(list(self.seq_name_to_sequence.values()))

        if self.calc_weights:  # Calculate sequence weights for sampling; pi_s = 1/(sum(I(dist(Xs, Xt) < theta))
            logger.info("Computing sequence weights")

            def weight_fn(x, theta):  # TODO: need to speed up distance calculation
                x_flat = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
                x_weights = 1.0 / np.sum(squareform(pdist(x_flat, metric="cosine")) < theta, axis=0)
                return x_weights

            weights = weight_fn(x_train, self.theta)
        else:  # If not calculate weights, set weights as ones
            weights = np.ones(x_train.shape[0])

        return x_train, x_train_names, weights


