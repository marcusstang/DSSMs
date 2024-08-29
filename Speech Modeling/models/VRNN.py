import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
from pyro.distributions import TransformedDistribution
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
    Adam,
    ClippedAdam,
)

from .utils import *

class VRNN(nn.Module):
    def __init__(
        self,
        dim_x=100,                # x dimensions
        dim_z=100,                # z dimensions
        dim_h=10,                 # dimensions of RNN hidden states in generative process
        dim_dx=20,                # hidden dimensions in network d_x
        dim_dz=20,                # hidden dimensions in network d_z
        dim_g=10,                 # dimensions of RNN hidden states in inference process
        num_layers=1,             # RNN layers
        rnn_dropout_rate=0.0,     # RNN dropout rate
        num_iafs=0,
        iaf_dim=50,
        use_cuda=False,
    ):
        super().__init__()
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        if use_cuda:
            self.cuda()

        self.dz = stoch_trans(dim_z, dim_x+dim_h, dim_dz)
        self.dx = stoch_trans(dim_x, dim_h, dim_dx)
        self.gen_RNN = nn.GRU(
            input_size = dim_x+dim_z,
            hidden_size = dim_h,
            #nonlinearity = "relu",
            batch_first = True,
            bidirectional = False,
            num_layers = num_layers,
            dropout = rnn_dropout_rate,
        )
        # bidirectional RNN in inference
        self.inf_RNN_bi = nn.GRU(
            input_size = dim_x,
            hidden_size = dim_g,
            #nonlinearity = "relu",
            batch_first = True,
            bidirectional = True,
            num_layers = num_layers,
            dropout = rnn_dropout_rate,
        )
        # backward RNN in inference
        self.inf_RNN_bw = nn.GRU(
            input_size = dim_x,
            hidden_size = dim_g,
            #nonlinearity = "relu",
            batch_first = True,
            bidirectional = False,
            num_layers = num_layers,
            dropout = rnn_dropout_rate,
        )
        self.combiner_1 = Combiner_1(dim_z, dim_g)
        self.combiner_2 = Combiner_2(dim_z, dim_h, dim_g)
        self.iafs = [
            affine_autoregressive(dim_z, hidden_dims=[iaf_dim]) for _ in range(num_iafs)
        ]
        self.iafs_modules = nn.ModuleList(self.iafs)
        self.h_0 = nn.Parameter(torch.zeros(1, 1, dim_h))
        self.z_q_0 = nn.Parameter(torch.zeros(dim_z))
        self.g_0_bi = nn.Parameter(torch.zeros(2, 1, dim_g))
        self.g_0_bw = nn.Parameter(torch.zeros(1, 1, dim_g))

    def model(self, mini_batch_x, annealing_factor=1.0):
        pyro.module("VRNN", self)
        mini_batch_x = mini_batch_x.to(self.device)
        T_max = mini_batch_x.size(1) # T
        batch_size = mini_batch_x.size(0) # batch size

        # p(z_0)
        x_prev = mini_batch_x[:, 0, :] # x_1
        h_prev = self.h_0.expand(1, batch_size, self.gen_RNN.hidden_size).contiguous() # h_1

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(2, T_max + 1)):
                # p(z_t|x_{t-1}, h_{t-1})
                dz_input = torch.cat((x_prev, h_prev[0,:,:]), dim=-1)
                z_loc, z_scale = self.dz(dz_input)
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t, dist.Normal(z_loc, z_scale).to_event(1))
                # ht = d_h(x_{t-1}, z_t, h_{t-1})
                dh_input = torch.cat((x_prev, z_t), dim=-1).unsqueeze(1)  # RNN expects input shaped like (N,L,H_in)
                _, h_prev = self.gen_RNN(dh_input, h_prev)
                # p(x_t|h_t)
                x_loc, x_scale = self.dx(h_prev[0,:,:])
                x_prev = pyro.sample("obs_x_%d" % t, dist.Normal(x_loc, x_scale).to_event(1), obs=mini_batch_x[:, t - 1, :])

    def guide_1(self, mini_batch_x, annealing_factor=1.0):
        pyro.module("VRNN", self)
        mini_batch_x = mini_batch_x.to(self.device)
        T_max = mini_batch_x.size(1) # T
        batch_size = mini_batch_x.size(0) # batch size
        H = self.inf_RNN_bi.hidden_size # hidden size, dim_g

        g_0_contig = self.g_0_bi.expand(2, batch_size, H).contiguous()
        rnn_output, _ = self.inf_RNN_bi(mini_batch_x, g_0_contig)

        # q(z_0)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(2, T_max + 1)):
                z_loc, z_scale = self.combiner_1(z_prev, rnn_output[:, t - 1, :H], rnn_output[:, t - 1, H:])
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                else:
                    z_dist = dist.Normal(z_loc, z_scale).to_event(1)
                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t, z_dist)
                # update time step
                z_prev = z_t

    def guide_2(self, mini_batch_x, annealing_factor=1.0):
        pyro.module("VRNN", self)
        mini_batch_x = mini_batch_x.to(self.device)
        T_max = mini_batch_x.size(1) # T
        batch_size = mini_batch_x.size(0) # batch size
        H = self.inf_RNN_bw.hidden_size # hidden size, dim_g

        g_0_contig = self.g_0_bw.expand(1, batch_size, H).contiguous()
        mini_batch_reversed = torch.flip(mini_batch_x, dims=[1])
        rnn_output, _ = self.inf_RNN_bw(mini_batch_reversed, g_0_contig)
        rnn_output = torch.flip(rnn_output, dims=[1])

        #x_prev = mini_batch_x[:, 0, :] # x_1
        h_prev = self.h_0.expand(1, batch_size, self.gen_RNN.hidden_size).contiguous() # h_1

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(2, T_max + 1)):
                z_loc, z_scale = self.combiner_2(h_prev[0, :, :], rnn_output[:, t - 2, :]) # note t-2 here
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                else:
                    z_dist = dist.Normal(z_loc, z_scale).to_event(1)
                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t, z_dist)
                # ht = d_h(x_{t-1}, z_t, h_{t-1})
                x_prev = mini_batch_x[:, t-2, :]
                #print("Sampled z_t shape:", z_t.shape)
                dh_input = torch.cat((x_prev, z_t), dim=-1).unsqueeze(1)
                _, h_prev = self.gen_RNN(dh_input, h_prev)
                # update time step
                z_prev = z_t


def reconstruct_vrnn(model_trained, x_data, guide_1=True, num_MC=1000):
    batch_size = x_data.shape[0] # batch size
    L = x_data.shape[1] # length
    H = model_trained.inf_RNN_bi.hidden_size # hidden size, dim_g

    h_prev = model_trained.h_0.expand(1, batch_size, model_trained.gen_RNN.hidden_size).contiguous()
    x1_loc, x1_scale = model_trained.dx(h_prev[0,:,:])

    x_pred = [x1_loc]
    x_scales = [x1_scale]

    if guide_1:
        g_0_contig = model_trained.g_0_bi.expand(2, batch_size, H).contiguous()
        rnn_output, _ = model_trained.inf_RNN_bi(x_data, g_0_contig)
        z_prev = model_trained.z_q_0.expand(batch_size, model_trained.z_q_0.size(0))
        for t in range(2,L+1):
            z_prev, z_scale = model_trained.combiner_1(z_prev, rnn_output[:, t - 1, :H], rnn_output[:, t - 1, H:])
            if len(model_trained.iafs) > 0:
                z_prev = mean_iaf(z_prev, z_scale, model_trained.iafs, num_samples = num_MC)
            dh_input = torch.cat((x_data[:,t-2,:], z_prev), dim=-1).unsqueeze(1)
            _, h_prev = model_trained.gen_RNN(dh_input, h_prev)
            x_prev, x_scale = model_trained.dx(h_prev[0,:,:])
            x_pred.append(x_prev)
            x_scales.append(x_scale)

    else:
        g_0_contig = model_trained.g_0_bw.expand(1, batch_size, H).contiguous()
        mini_batch_reversed = torch.flip(x_data, dims=[1])
        rnn_output, _ = model_trained.inf_RNN_bw(mini_batch_reversed, g_0_contig)
        rnn_output = torch.flip(rnn_output, dims=[1])
        for t in range(2,L+1):
            z_prev, z_scale = model_trained.combiner_2(h_prev[0, :, :], rnn_output[:, t - 2, :])
            if len(model_trained.iafs) > 0:
                z_prev = mean_iaf(z_prev, z_scale, model_trained.iafs, num_samples = num_MC)
            dh_input = torch.cat((x_data[:,t-2,:], z_prev), dim=-1).unsqueeze(1)
            _, h_prev = model_trained.gen_RNN(dh_input, h_prev)
            x_prev, x_scale = model_trained.dx(h_prev[0,:,:])
            x_pred.append(x_prev)
            x_scales.append(x_scale)

    x_pred = torch.stack(x_pred, dim=1).detach().numpy()
    x_scales = torch.stack(x_scales, dim=1).detach().numpy()
    return x_pred, x_scales
