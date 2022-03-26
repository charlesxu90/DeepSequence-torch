"""
Adapted from: https://github.com/debbiemarkslab/DeepSequence/blob/master/DeepSequence/helper.py
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
import logging
from fastdist import fastdist

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_path(base_dir, base_name, suffix):
    return os.path.join(base_dir, base_name + suffix)

class DataHelper:
    def __init__(self, alignment_file="", focus_seq_name="", calc_weights=True, working_dir=".",
                 theta=0.2, load_all_sequences=True) -> object:

        """
        Class to load and organize alignment data. This function also helps makes predictions about mutations.

        Params:
        --------------
        alignment_file:     Path to the alignment file
        focus_seq_name:     Name of the sequence in the alignment. Default: the first sequence in the alignment
        calc_weights:       Calculate sequence weights or not. Default: True
        working_dir:        Directory to save "params", "logs", and "embeddings"
        theta:              Sequence weighting hyperparameter. Default=0.2
                            Generally: 0.2 for prokaryotic and eukaryotic, and 0.01 for Viruses.
        load_all_sequences: Default = False
        """
        np.random.seed(42)
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.working_dir = working_dir
        self.calc_weights = calc_weights

        self.wt_elbo = None  # Needed for mutation effect prediction
        self.theta = theta  # Alignment processing parameters
        self.load_all_sequences = load_all_sequences  # Testing, no need for all sequence loaded

        # only protein supported
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.reorder_alphabet = "DEKRHNQSTPGAVILMCFYW"
        self.alphabet_size = len(self.alphabet)

        self.aa_dict = {aa: i for i, aa in enumerate(self.alphabet)}  # init alphabet dictionary, aa to index
        self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}  # index to aa

        self.seq_name_to_sequence, self.focus_cols = self.gen_basic_alignment()  # generate the experimental data
        self.seq_len = len(self.focus_cols)

        if self.load_all_sequences:
            self.x_train, self.weights, self.Neff = self.gen_full_alignment()

    def read_alignment_file(self):
        seq_names = []
        seq_name_to_sequence = {}

        name = ""
        INPUT = open(self.alignment_file, "r")
        for i, line in enumerate(INPUT):
            line = line.rstrip()
            if line.startswith(">"):
                name = line
                seq_names.append(name)
            else:
                if name in seq_name_to_sequence.keys():
                    seq_name_to_sequence[name] += line
                else:
                    seq_name_to_sequence[name] = line
        INPUT.close()

        return seq_names, seq_name_to_sequence

    def gen_basic_alignment(self):
        """ Read training alignment and store basics in class instance """
        seq_names, seq_name_to_sequence = self.read_alignment_file()
        logger.info(f"Totally {len(seq_name_to_sequence)} in the alignment.")

        if self.focus_seq_name == "":  # pick the first seq if no focus sequence
            self.focus_seq_name = seq_names[0]

        # Select focus columns These columns are the uppercase residues of the .a2m file
        focus_seq = seq_name_to_sequence[self.focus_seq_name]
        focus_cols = [ix for ix, s in enumerate(focus_seq) if s == s.upper()]
        focus_seq_trimmed = [focus_seq[ix] for ix in focus_cols]

        # The focus sequence is expected to be named as: >[NAME]/[start]-[end], need to modify if needed.
        focus_loc = self.focus_seq_name.split("/")[-1]
        focus_start, focus_stop = (int(pos) for pos in focus_loc.split("-"))

        uniprot_focus_cols_list = [idx_col+focus_start for idx_col in focus_cols]
        uniprot_focus_col_to_wt_aa_dict = {idx_col+focus_start: focus_seq[idx_col] for idx_col in focus_cols}
        uniprot_focus_col_to_focus_idx = {idx_col+focus_start: idx_col for idx_col in focus_cols}
        return seq_name_to_sequence, focus_cols

    def remove_invalid_seqs(self):
        alphabet_set = set(list(self.alphabet))  # Remove sequences that have bad characters
        seq_names_to_remove = []
        for seq_name, sequence in self.seq_name_to_sequence.items():
            for letter in sequence:
                if letter not in alphabet_set and letter != "-":
                    seq_names_to_remove.append(seq_name)

        for seq_name in set(seq_names_to_remove):  # remove invalid seqs
            del self.seq_name_to_sequence[seq_name]

        logger.info(f"{len(self.seq_name_to_sequence)} sequences remains after removing invalid!")

    def encode_alignment_seqs(self):  # Encode the sequences
        x_train = np.zeros((len(self.seq_name_to_sequence.keys()), len(self.focus_cols), len(self.alphabet)))
        x_train_name_list = []
        for i, seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence[seq_name]
            x_train_name_list.append(seq_name)
            for j, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    k = self.aa_dict[letter]
                    x_train[i, j, k] = 1.0
        return x_train, x_train_name_list

    def gen_full_alignment(self):

        # Get only the focus columns
        for seq_name, sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".", "-")  # Replace periods with dashes (the uppercase equivalent)
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]  # only use focus cols

        self.remove_invalid_seqs()  # remove invalid seqs in self.seq_name_to_sequence
        x_train, x_train_name_list = self.encode_alignment_seqs()

        if self.calc_weights:  # Calculate sequence weights for sampling; pi_s = 1/(sum(I(dist(Xs, Xt) < theta))
            logger.info("Computing sequence weights")

            def weight_fn(x, theta):
                x_flat = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
                x_weights = 1.0 / np.sum(squareform(pdist(x_flat, metric="cosine")) < theta, axis=0)
                return x_weights

            weights = weight_fn(x_train, self.theta)
        else:  # If not calculate weights, set weights as ones
            weights = np.ones(x_train.shape[0])

        neff = np.sum(weights)
        logger.info(f"neff = {neff}, Data Shape = {x_train.shape}")
        return x_train, weights, neff

    def one_hot_3D(self, s):
        """ Transform sequence string into one-hot aa vector"""
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i, self.aa_dict[letter]] = 1
        return x

    def delta_elbo(self, model, mutant_tuple_list, N_pred_iterations=10):

        for pos,wt_aa,mut_aa in mutant_tuple_list:
            if pos not in self.uniprot_focus_col_to_wt_aa_dict \
                or self.uniprot_focus_col_to_wt_aa_dict[pos] != wt_aa:
                print ("Not a valid mutant!",pos,wt_aa,mut_aa)
                return None

        mut_seq = self.focus_seq_trimmed[:]
        for pos,wt_aa,mut_aa in mutant_tuple_list:
            mut_seq[self.uniprot_focus_col_to_focus_idx[pos]] = mut_aa


        if self.wt_elbo == None:
            mutant_sequences = [self.focus_seq_trimmed, mut_seq]
        else:
            mutant_sequences = [mut_seq]

        # Then make the one hot sequence
        mutant_sequences_one_hot = np.zeros(\
            (len(mutant_sequences),len(self.focus_cols),len(self.alphabet)))

        for i,sequence in enumerate(mutant_sequences):
            for j,letter in enumerate(sequence):
                k = self.aa_dict[letter]
                mutant_sequences_one_hot[i,j,k] = 1.0

        prediction_matrix = np.zeros((mutant_sequences_one_hot.shape[0],N_pred_iterations))
        idx_batch = np.arange(mutant_sequences_one_hot.shape[0])
        for i in range(N_pred_iterations):

            batch_preds, _, _ = model.all_likelihood_components(mutant_sequences_one_hot)

            prediction_matrix[:,i] = batch_preds

        # Then take the mean of all my elbo samples
        mean_elbos = np.mean(prediction_matrix, axis=1).flatten().tolist()

        if self.wt_elbo == None:
            self.wt_elbo = mean_elbos.pop(0)

        return mean_elbos[0] - self.wt_elbo

    def single_mutant_matrix(self, model, N_pred_iterations=10, minibatch_size=2000, filename_prefix=""):

        """ Predict the delta elbo for all single mutants """

        # Get the start and end index from the sequence name
        start_idx, end_idx = self.focus_seq_name.split("/")[-1].split("-")
        start_idx = int(start_idx)

        wt_pos_focus_idx_tuple_list = []
        focus_seq_index = 0
        focus_seq_list = []
        for i,letter in enumerate(self.focus_seq):
            if letter == letter.upper():
                wt_pos_focus_idx_tuple_list.append((letter,start_idx+i,focus_seq_index))
                focus_seq_index += 1

        self.mutant_sequences = ["".join(self.focus_seq_trimmed)]
        self.mutant_sequences_descriptor = ["wt"]
        for wt,pos,idx_focus in wt_pos_focus_idx_tuple_list:
            for mut in self.alphabet:
                if wt != mut:
                    # Make a descriptor
                    descriptor = wt+str(pos)+mut

                    # Hard copy the sequence
                    focus_seq_copy = list(self.focus_seq_trimmed)[:]

                    # Mutate
                    focus_seq_copy[idx_focus] = mut

                    # Add to the list
                    self.mutant_sequences.append("".join(focus_seq_copy))
                    self.mutant_sequences_descriptor.append(descriptor)

        # Then make the one hot sequence
        self.mutant_sequences_one_hot = np.zeros(\
            (len(self.mutant_sequences),len(self.focus_cols),len(self.alphabet)))

        for i,sequence in enumerate(self.mutant_sequences):
            for j,letter in enumerate(sequence):
                k = self.aa_dict[letter]
                self.mutant_sequences_one_hot[i,j,k] = 1.0

        self.prediction_matrix = np.zeros((self.mutant_sequences_one_hot.shape[0],N_pred_iterations))

        batch_order = np.arange(self.mutant_sequences_one_hot.shape[0])

        for i in range(N_pred_iterations):
            np.random.shuffle(batch_order)

            for j in range(0,self.mutant_sequences_one_hot.shape[0],minibatch_size):

                batch_index = batch_order[j:j+minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.mutant_sequences_one_hot[batch_index])

                for k,idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i]= batch_preds[k]

        # Then take the mean of all my elbo samples
        self.mean_elbos = np.mean(self.prediction_matrix, axis=1).flatten().tolist()

        self.wt_elbo = self.mean_elbos.pop(0)
        self.mutant_sequences_descriptor.pop(0)

        self.delta_elbos = np.asarray(self.mean_elbos) - self.wt_elbo

        if filename_prefix == "":
            return self.mutant_sequences_descriptor, self.delta_elbos

        else:
            OUTPUT = open(filename_prefix+"_samples-"+str(N_pred_iterations)\
                +"_elbo_predictions.csv", "w")

            for i,descriptor in enumerate(self.mutant_sequences_descriptor):
                OUTPUT.write(descriptor+";"+str(self.mean_elbos[i])+"\n")

            OUTPUT.close()

    def custom_mutant_matrix(self, input_filename, model, N_pred_iterations=10, minibatch_size=2000,
                             filename_prefix="", offset=0):

        """ Predict the delta elbo for a custom mutation filename
        """
        # Get the start and end index from the sequence name
        start_idx, end_idx = self.focus_seq_name.split("/")[-1].split("-")
        start_idx = int(start_idx)

        wt_pos_focus_idx_tuple_list = []
        focus_seq_index = 0
        focus_seq_list = []
        mutant_to_letter_pos_idx_focus_list = {}

        # find all possible valid mutations that can be run with this alignment
        for i,letter in enumerate(self.focus_seq):
            if letter == letter.upper():
                for mut in self.alphabet:
                    pos = start_idx+i
                    if letter != mut:
                        mutant = letter+str(pos)+mut
                        mutant_to_letter_pos_idx_focus_list[mutant] = [letter,start_idx+i,focus_seq_index]
                focus_seq_index += 1

        self.mutant_sequences = ["".join(self.focus_seq_trimmed)]
        self.mutant_sequences_descriptor = ["wt"]

        # run through the input file
        INPUT = open(self.working_dir+"/"+input_filename, "r")
        for i,line in enumerate(INPUT):
            line = line.rstrip()
            if i >= 1:
                line_list = line.split(",")
                # generate the list of mutants
                mutant_list = line_list[0].split(":")
                valid_mutant = True

                # if any of the mutants in this list aren"t in the focus sequence,
                #    I cannot make a prediction
                for mutant in mutant_list:
                    if mutant not in mutant_to_letter_pos_idx_focus_list:
                        valid_mutant = False

                # If it is a valid mutant, add it to my list to make preditions
                if valid_mutant:
                    focus_seq_copy = list(self.focus_seq_trimmed)[:]

                    for mutant in mutant_list:
                        wt_aa,pos,idx_focus = mutant_to_letter_pos_idx_focus_list[mutant]
                        mut_aa = mutant[-1]
                        focus_seq_copy[idx_focus] = mut_aa

                    self.mutant_sequences.append("".join(focus_seq_copy))
                    self.mutant_sequences_descriptor.append(":".join(mutant_list))

        INPUT.close()

        # Then make the one hot sequence
        self.mutant_sequences_one_hot = np.zeros(\
            (len(self.mutant_sequences),len(self.focus_cols),len(self.alphabet)))

        for i,sequence in enumerate(self.mutant_sequences):
            for j,letter in enumerate(sequence):
                k = self.aa_dict[letter]
                self.mutant_sequences_one_hot[i,j,k] = 1.0

        self.prediction_matrix = np.zeros((self.mutant_sequences_one_hot.shape[0],N_pred_iterations))

        batch_order = np.arange(self.mutant_sequences_one_hot.shape[0])

        for i in range(N_pred_iterations):
            np.random.shuffle(batch_order)

            for j in range(0,self.mutant_sequences_one_hot.shape[0],minibatch_size):

                batch_index = batch_order[j:j+minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.mutant_sequences_one_hot[batch_index])

                for k,idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i]= batch_preds[k]

        # Then take the mean of all my elbo samples
        self.mean_elbos = np.mean(self.prediction_matrix, axis=1).flatten().tolist()

        self.wt_elbo = self.mean_elbos.pop(0)
        self.mutant_sequences_descriptor.pop(0)

        self.delta_elbos = np.asarray(self.mean_elbos) - self.wt_elbo

        if filename_prefix == "":
            return self.mutant_sequences_descriptor, self.delta_elbos

        else:

            OUTPUT = open(filename_prefix+"_samples-"+str(N_pred_iterations)\
                +"_elbo_predictions.csv", "w")

            for i,descriptor in enumerate(self.mutant_sequences_descriptor):
                OUTPUT.write(descriptor+";"+str(self.delta_elbos[i])+"\n")

            OUTPUT.close()

    def get_pattern_activations(self, model, update_num, filename_prefix="", verbose=False, minibatch_size=2000):

        activations_filename = self.working_dir+"/embeddings/"+filename_prefix+"_pattern_activations.csv"

        OUTPUT = open(activations_filename, "w")

        batch_order = np.arange(len(self.x_train_name_list))

        for i in range(0,len(self.x_train_name_list),minibatch_size):
            batch_index = batch_order[i:i+minibatch_size]
            one_hot_seqs = self.x_train[batch_index]
            batch_activation = model.get_pattern_activations(one_hot_seqs)

            for j,idx in enumerate(batch_index.tolist()):
                sample_activation = [str(val) for val in batch_activation[j].tolist()]
                sample_name = self.x_train_name_list[idx]
                out_line = [str(update_num),sample_name]+sample_activation
                if verbose:
                    print ("\t".join(out_line))
                OUTPUT.write(",".join(out_line)+"\n")

        OUTPUT.close()

    def get_embeddings(self, model, update_num, filename_prefix="",
                        verbose=False, minibatch_size=2000):
        """ Save the latent variables from all the sequences in the alignment """
        embedding_filename = self.working_dir+"/embeddings/"+filename_prefix+"_seq_embeddings.csv"

        # Append embeddings to file if it has already been created
        #   This is useful if you want to see the embeddings evolve over time
        if os.path.isfile(embedding_filename):
            OUTPUT = open(embedding_filename, "a")

        else:
            OUTPUT = open(embedding_filename, "w")
            mu_header_list = ["mu_"+str(i+1) for i in range(model.n_latent)]
            log_sigma_header_list = ["log_sigma_"+str(i+1) for i in range(model.n_latent)]

            header_list = mu_header_list + log_sigma_header_list
            OUTPUT.write("update_num,name,"+",".join(header_list)+"\n")


        batch_order = np.arange(len(self.x_train_name_list))

        for i in range(0,len(self.x_train_name_list),minibatch_size):
            batch_index = batch_order[i:i+minibatch_size]
            one_hot_seqs = self.x_train[batch_index]
            batch_mu, batch_log_sigma  = model.recognize(one_hot_seqs)

            for j,idx in enumerate(batch_index.tolist()):
                sample_mu = [str(val) for val in batch_mu[j].tolist()]
                sample_log_sigma = [str(val) for val in batch_log_sigma[j].tolist()]
                sample_name = self.x_train_name_list[idx]
                out_line = [str(update_num),sample_name]+sample_mu+sample_log_sigma
                if verbose:
                    print ("\t".join(out_line))
                OUTPUT.write(",".join(out_line)+"\n")

        OUTPUT.close()

    def get_elbo_samples(self, model, N_pred_iterations=100, minibatch_size=2000):

        self.prediction_matrix = np.zeros((self.one_hot_mut_array_with_wt.shape[0],N_pred_iterations))

        batch_order = np.arange(self.one_hot_mut_array_with_wt.shape[0])

        for i in range(N_pred_iterations):
            np.random.shuffle(batch_order)

            for j in range(0,self.one_hot_mut_array_with_wt.shape[0],minibatch_size):

                batch_index = batch_order[j:j+minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.one_hot_mut_array_with_wt[batch_index])

                for k,idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i]= batch_preds[k]

