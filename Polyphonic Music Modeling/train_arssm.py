from models.arssm import *
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
    JitTrace_ELBO,
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
    model_name = f"arssm_{dataset_name.lower()}_epoch_{epoch}.pth"
    model_path = os.path.join(checkpoint_dir, model_name)
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{model_name}")

    # Save model state
    logging.info(f"Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)

    # Save optimizer state using Pyro's save method
    logging.info(f"Saving optimizer states to {optimizer_path}...")
    optimizer.save(optimizer_path)

    logging.info("Done saving model and optimizer checkpoints to disk.")

def load_checkpoint(model, optimizer, dataset_name, checkpoint_dir="checkpoints"):
    # Find the latest checkpoint for the dataset
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"arssm_{dataset_name.lower()}")]
    if not model_files:
        logging.info("No checkpoints found. Starting from scratch.")
        return 0

    # Sort checkpoints by epoch number (assuming filenames follow the correct format)
    model_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    latest_model_file = model_files[-1]
    epoch = int(latest_model_file.split('_')[-1].split('.')[0])

    model_path = os.path.join(checkpoint_dir, latest_model_file)
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{latest_model_file}")

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
    # dataset_name = "NOTTINGHAM"
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

    # training set
    training_seq_lengths = data["train"]["sequence_lengths"]  # [229]
    training_data_sequences = data["train"]["sequences"]  # [229, 129, 88] # there are 88 kinds of notes

    # number of training sequences
    N_train_data = len(training_seq_lengths)  # 229
    # total length of traning data
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    # number of mini-batches
    N_mini_batches = int(N_train_data / args.mini_batch_size + int(N_train_data % args.mini_batch_size > 0))  # 12 batches

    logging.info(
        "N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d"
        % (N_train_data, training_seq_lengths.float().mean(), N_mini_batches)
    )


    # ------------------------------ training ------------------------------
    pyro.set_rng_seed(0)
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
        "lr": 0.0003,  # 0.0003,
        "betas": (0.96, 0.999),
        "clip_norm": 20.0,  # 10.0,
        "lrd": 0.99996,
        "weight_decay": 2.0,
    }

    adam = ClippedAdam(adam_params)
    svi = SVI(arssm.model, arssm.guide_2, adam, Trace_ELBO())

    # Load checkpoint if it exists
    start_epoch = load_checkpoint(arssm, adam, dataset_name)

    def process_minibatch(epoch, which_mini_batch, shuffled_indices):
        if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
            min_af = args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * (
                    float(which_mini_batch + epoch * N_mini_batches + 1)
                    / float(args.annealing_epochs * N_mini_batches)
            )
        else:
            annealing_factor = 1.0

        # compute which sequences in the training set we should grab
        mini_batch_start = which_mini_batch * args.mini_batch_size
        mini_batch_end = np.min([(which_mini_batch + 1) * args.mini_batch_size, N_train_data])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
        # grab a fully prepped mini-batch using the helper function in the data loader
        (
            mini_batch,
            mini_batch_reversed,
            mini_batch_mask,
            mini_batch_seq_lengths,
        ) = poly.get_mini_batch(
            mini_batch_indices,
            training_data_sequences,
            training_seq_lengths,
            cuda=args.cuda,
        )
        # do an actual gradient step
        loss = svi.step(
            mini_batch,
            mini_batch_reversed,
            mini_batch_mask,
            mini_batch_seq_lengths,
            annealing_factor,
        )
        # keep track of the training loss
        return loss

    for epoch in range(start_epoch, args.num_epochs):
        epoch_nll = 0.0
        # prepare mini-batch subsampling indices for this epoch
        shuffled_indices = torch.randperm(N_train_data)

        # process each mini-batch; this is where we take gradient steps
        i = 1
        for which_mini_batch in range(N_mini_batches):
            epoch_nll += process_minibatch(epoch, which_mini_batch, shuffled_indices)
            # print(i)
            i += 1

        # report training diagnostics
        print("[training epoch %04d]  %.4f \t\t\t\t" % (epoch, epoch_nll / N_train_time_slices))


        # Save the model every 100 epochs
        if epoch > 0 and epoch % 20 == 0:
            save_checkpoint(arssm, adam, epoch, dataset_name)


if __name__ == '__main__':
    main()
