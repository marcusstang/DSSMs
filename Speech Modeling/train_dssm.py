import os
import datetime

import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import SpeechSequencesFull
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
#from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceTMC_ELBO,
    config_enumerate,
)
from pyro.optim import (
    DCTAdam,
    ClippedAdam,
)

from misc.gen_data import *
from models.utils import *
from models.VRNN import *
from models.FSSM import *
from models.ArFSSM import *
from models.RSSM import *
from models.SRNN import *

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    save_dir = "./results/arfssm_2"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    torch.manual_seed(42)
    np.random.seed(torch.initial_seed() % (2**32 - 1))

    wlen_sec = 64e-3
    hop_percent = 0.25
    fs =  16e3
    zp_percent = 0
    trim = True
    seq_len = 50
    batch_size = 100

    wlen = wlen_sec * fs
    wlen = np.int64(np.power(2, np.ceil(np.log2(wlen))))  # pwoer of 2
    hop = np.int64(hop_percent * wlen)
    nfft = wlen + zp_percent * wlen
    win = torch.sin(torch.arange(0.5, wlen + 0.5) / wlen * np.pi)

    STFT_dict = {}
    STFT_dict['fs'] = fs
    STFT_dict['wlen'] = wlen
    STFT_dict['hop'] = hop
    STFT_dict['nfft'] = nfft
    STFT_dict['win'] = win
    STFT_dict['trim'] = trim

    data_suffix = "flac"

    train_file_list = librosa.util.find_files('./data/train', ext=data_suffix)
    train_dataset = SpeechSequencesFull(file_list=train_file_list, sequence_len=seq_len,
                                        STFT_dict=STFT_dict, shuffle=True)
    val_file_list = librosa.util.find_files('./data/val', ext=data_suffix)
    val_dataset = SpeechSequencesFull(file_list=val_file_list, sequence_len=seq_len,
                                        STFT_dict=STFT_dict, shuffle=True)

    train_num = train_dataset.__len__()
    val_num = val_dataset.__len__()
    print('Training samples: {}'.format(train_num))
    print('Validating samples: {}'.format(val_num))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

    srnn = ArFSSM(
        dim_x=513,
        dim_z=16,
        dim_h1=16,
        dim_h2=128,
        dim_dx=256,
        dim_dz=32,
        dim_g=128,
        num_layers=1,
        rnn_dropout_rate=0.0,
    )

    '''save_file = os.path.join('./results/vrnn' + str(269) + '.pt')
    vrnn.load_state_dict(torch.load(save_file, map_location='cpu'))'''

    adam_params = {
        "lr": 0.0008,  # 0.0003,
        "betas": (0.96, 0.999),
        "clip_norm": 10.0,  # 10.0,
        "lrd": 0.99996,
        "weight_decay": 2.0,
    }
    adam = ClippedAdam(adam_params)
    svi = SVI(srnn.model, srnn.guide_2, adam, Trace_ELBO())

    NUM_EPOCH = 1000
    annealing_epochs = 100
    minimum_annealing_factor = 0.001
    acc_train_loss = np.zeros(NUM_EPOCH)
    acc_val_loss = np.zeros(NUM_EPOCH)
    best_loss = float("inf")
    check_intervals = 10
    NUM_PATIENCE = 50
    patience = NUM_PATIENCE
    for epoch in range(NUM_EPOCH):

        start_time = datetime.datetime.now()

        ## training
        for which_mini_batch, batch_data in enumerate(train_dataloader):
            if annealing_epochs > 0 and epoch < annealing_epochs:
                annealing_factor = minimum_annealing_factor + (1.0 - minimum_annealing_factor) * (
                        float(which_mini_batch + epoch * batch_size + 1)
                        / float(annealing_epochs * batch_size)
                )
            else:
                annealing_factor = 1.0

            batch_data = batch_data.to(device).permute(0, 2, 1)

            loss = svi.step(batch_data, annealing_factor) // seq_len
            acc_train_loss[epoch] += loss

        # validating
        for which_mini_batch, batch_data in enumerate(val_dataloader):
            if annealing_epochs > 0 and epoch < annealing_epochs:
                annealing_factor = minimum_annealing_factor + (1.0 - minimum_annealing_factor) * (
                        float(which_mini_batch + epoch * batch_size + 1)
                        / float(annealing_epochs * batch_size)
                )
            else:
                annealing_factor = 1.0

            batch_data = batch_data.to(device).permute(0, 2, 1)

            loss = svi.evaluate_loss(batch_data, annealing_factor) // seq_len
            acc_val_loss[epoch] += loss

        acc_train_loss[epoch] = acc_train_loss[epoch] / train_num
        acc_val_loss[epoch] = acc_val_loss[epoch] / train_num

        end_time = datetime.datetime.now()
        interval = (end_time - start_time).seconds / 60
        print('Epoch: {} training time {:.2f}m'.format(epoch + 1, interval))
        print('Train => tot: {:.2f}, Validate => tot: {:.2f}'.format(acc_train_loss[epoch], acc_val_loss[epoch]))

        '''if acc_val_loss[epoch] < best_loss:
            patience = NUM_PATIENCE
        else:
            patience -= 1
            if patience == 0:
                break'''

        if (epoch + 1) % check_intervals == 0:
            save_file = os.path.join(save_dir, 'arfssm' + str(epoch + 1) + '.pt')
            torch.save(srnn.state_dict(), save_file)
            if acc_val_loss[epoch] < best_loss:
                best_loss = acc_val_loss[epoch]
                save_file = os.path.join(save_dir, 'best.pt')
                torch.save(srnn.state_dict(), save_file)

def main():
    train()
    print(666)

if __name__ == '__main__':
    main()