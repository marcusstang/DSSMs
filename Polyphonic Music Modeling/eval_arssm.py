from models.arssm import ArFSSM
from utils import *

import os
import logging
import pickle
import numpy as np
import torch
import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
from pyro.infer import (
    SVI,
    Trace_ELBO,
)
from pyro.optim import (
    Adam,
    ClippedAdam,
)


datasets = {
    "JSB_CHORALES": poly.JSB_CHORALES,
    "MUSE_DATA": poly.MUSE_DATA,
    "NOTTINGHAM": poly.NOTTINGHAM,
    "PIANO_MIDI": poly.PIANO_MIDI
}

def load_dataset(dataset_name):
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not recognized. Available datasets: {list(datasets.keys())}")

    processed_data_path = f"C:\\Users\\sucra\\Desktop\\DSSM_music\\.venv\\Lib\\site-packages\\pyro\\contrib\\examples\\.data\\{dataset_name.lower()}_data.pkl"

    if os.path.exists(processed_data_path):
        print(f"Loading processed data for {dataset_name} from {processed_data_path}")
        with open(processed_data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print(f"Processing and dumping data for {dataset_name} to {processed_data_path}")
        data = poly.load_data(datasets[dataset_name])
        with open(processed_data_path, 'wb') as f:
            pickle.dump(data, f)

    return data

def save_checkpoint(model, optimizer, epoch, dataset_name, checkpoint_dir="checkpoints"):
    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Paths for saving the model and optimizer states
    model_name = f"vrnn_{dataset_name.lower()}_epoch_{epoch}.pth"
    model_path = os.path.join(checkpoint_dir, model_name)
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{model_name}")

    # Save model state
    logging.info(f"Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)

    # Save optimizer state using Pyro's save method
    logging.info(f"Saving optimizer states to {optimizer_path}...")
    optimizer.save(optimizer_path)

    logging.info("Done saving model and optimizer checkpoints to disk.")

def load_checkpoint(model, optimizer, dataset_name, checkpoint_dir="checkpoints", epoch=None):
    # Find all checkpoints for the dataset
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"arssm_{dataset_name.lower()}")]
    if not model_files:
        logging.info("No checkpoints found. Starting from scratch.")
        return 0

    # Sort checkpoints by epoch number (assuming filenames follow the correct format)
    model_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

    if epoch is None:
        # Load the latest checkpoint if no epoch is specified
        model_file = model_files[-1]
        epoch = int(model_file.split('_')[-1].split('.')[0])
    else:
        # Find the checkpoint for the specified epoch
        model_file = f"arssm_{dataset_name.lower()}_epoch_{epoch}.pth"
        if model_file not in model_files:
            logging.error(f"Checkpoint for epoch {epoch} not found. Available epochs: {[int(f.split('_')[-1].split('.')[0]) for f in model_files]}")
            return 0

    model_path = os.path.join(checkpoint_dir, model_file)
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{model_file}")

    # Load model state
    logging.info(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path))

    # Load optimizer state using Pyro's load method
    logging.info(f"Loading optimizer states from {optimizer_path}...")
    optimizer.load(optimizer_path)

    logging.info("Done loading model and optimizer states.")
    return epoch


def main():
    #------------------------------ Loading data ------------------------------
    dataset_name = "JSB_CHORALES"
    #dataset_name = "MUSE_DATA"
    #dataset_name = "NOTTINGHAM"
    #dataset_name = "PIANO_MIDI"

    data = load_dataset(dataset_name)

    class Args:
        def __init__(self):
            self.num_epochs = 1001
            self.mini_batch_size = 20
            self.annealing_epochs = 100
            self.minimum_annealing_factor = 0.2
            self.cuda = False

    args = Args()

    # test set
    test_seq_lengths = data["test"]["sequence_lengths"]  # [77]
    test_data_sequences = data["test"]["sequences"]  # [77, 160, 88]

    # number of testing sequences
    N_test_data = len(test_seq_lengths)  # 77
    # the number of samples we use to do the evaluation
    n_eval_samples = 1

    def rep_data(x):
        # Repeat the data n_eval_samples times along the first dimension
        repeat_dims = [n_eval_samples] + [1] * (x.dim() - 1)
        # Reshape after repeating to match the original dimensions with added repetitions
        rep_shape = (x.size(0) * n_eval_samples,) + x.size()[1:]
        return x.repeat(*repeat_dims).reshape(rep_shape)


    def rep_seq_lengths(x):
        return x.repeat(n_eval_samples)

    # Convert and replicate sequence lengths properly
    if not isinstance(test_seq_lengths, torch.Tensor):
        test_seq_lengths = torch.tensor(test_seq_lengths)
    else:
        test_seq_lengths = test_seq_lengths.clone().detach()

    #test_seq_lengths = rep_seq_lengths(test_seq_lengths)
    test_data_sequences = rep_data(test_data_sequences)
    test_seq_lengths = rep_seq_lengths(test_seq_lengths)

    # Get the entire test data ready for evaluation: pack into sequences, etc.
    (
        test_batch,
        test_batch_reversed,
        test_batch_mask,
        test_seq_lengths,
    ) = poly.get_mini_batch(
        #torch.arange(n_eval_samples * N_test_data),
        torch.arange(test_data_sequences.shape[0]),
        #rep_data(test_data_sequences),
        test_data_sequences,
        test_seq_lengths,
        cuda=args.cuda,
    )

    # ------------------------------ debugging ------------------------------
    # training set
    training_seq_lengths = data["train"]["sequence_lengths"]  # [229]
    training_data_sequences = data["train"]["sequences"]  # [229, 129, 88] # there are 88 kinds of notes
    N_train_data = len(training_seq_lengths)  # 229
    (
        train_batch,
        train_batch_reversed,
        train_batch_mask,
        train_seq_lengths,
    ) = poly.get_mini_batch(
        torch.arange(training_data_sequences.shape[0]),  # Use the entire test set without subsampling
        training_data_sequences,
        training_seq_lengths,
        cuda=args.cuda,
    )

    # ------------------------------ evaluation ------------------------------
    pyro.set_rng_seed(2)
    pyro.clear_param_store()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arssm = ArFSSM(
        dim_x=88,  # x dimensions
        dim_z=100,
        dim_h1=200,
        dim_h2=200,  # dimensions of RNN hidden states in generative process,
        dim_dx=100,  # hidden dimensions in network d_x
        dim_dz=100,  # hidden dimensions in network d_z
        dim_g=600,  # dimensions of RNN hidden states in inference process
        num_layers=1,  # RNN layers
        rnn_dropout_rate=0.0,  # RNN dropout rate
        num_iafs=0,
        iaf_dim=100,
        use_cuda=True,
    ).to(device)

    adam_params = {
        "lr": 0.0008,  # 0.0003,
        "betas": (0.96, 0.999),
        "clip_norm": 20.0,  # 10.0,
        "lrd": 0.99996,
        "weight_decay": 2.0,
    }

    adam = ClippedAdam(adam_params)
    #svi = SVI(arssm.model, arssm.guide_1, adam, Trace_ELBO())
    svi = SVI(arssm.model, arssm.guide_2, adam, TraceELBONLL())

    # Load checkpoint if it exists
    start_epoch = load_checkpoint(arssm, adam, dataset_name, epoch=920)



    arssm.gen_RNN_1.eval()
    arssm.gen_RNN_2.eval()
    arssm.inf_RNN_bi.eval()
    arssm.inf_RNN_bw.eval()

    #test_nll = get_nll(vrnn.model, vrnn.guide_1, test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths, num_samples=10)
    #print(test_nll)

    #elbo = get_elbo(vrnn.model, vrnn.guide_1, train_batch, train_batch_reversed, train_batch_mask, train_seq_lengths, num_samples=1)
    #print(elbo)

    def do_evaluation():
        # put the RNN into evaluation mode (i.e. turn off drop-out if applicable)

        # compute the validation and test loss n_samples many times
        train_nll = svi.evaluate_loss(
            train_batch, train_batch_reversed, train_batch_mask, train_seq_lengths,
        ) / float(torch.sum(train_seq_lengths))
        test_nll = svi.evaluate_loss(
            test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths
        ) / float(torch.sum(test_seq_lengths))

        # put the RNN back into training mode (i.e. turn on drop-out if applicable)
        return train_nll, test_nll

    train_nll, test_nll = do_evaluation()
    print(train_nll, test_nll)

if __name__ == '__main__':
    main()
