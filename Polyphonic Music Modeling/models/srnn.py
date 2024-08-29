from .utils import *

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pyro
import pyro.poutine as poutine
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.distributions as dist
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive

class SRNN(nn.Module):
    def __init__(
        self,
        dim_x=100,                # x dimensions
        dim_z=100,                # z dimensions
        dim_h=10,                 # dimensions of h in generative process
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

        self.dz = stoch_trans(dim_z, dim_z+dim_h, dim_dz)
        self.dx = emitter(dim_x, dim_z+dim_h, dim_dx)
        self.gen_RNN = nn.GRU(
            input_size = dim_x,
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
        self.combiner_3 = Combiner_3(dim_z, dim_z, dim_h, dim_g)
        self.iafs = [
            affine_autoregressive(dim_z, hidden_dims=[iaf_dim]) for _ in range(num_iafs)
        ]
        self.iafs_modules = nn.ModuleList(self.iafs)
        self.h_0 = nn.Parameter(torch.zeros(1, 1, dim_h))
        self.z_0 = nn.Parameter(torch.zeros(dim_z))
        self.z_q_0 = nn.Parameter(torch.zeros(dim_z))
        self.g_0_bi = nn.Parameter(torch.zeros(2, 1, dim_g))
        self.g_0_bw = nn.Parameter(torch.zeros(1, 1, dim_g))

    def model(self, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, annealing_factor=1.0):
        pyro.module("SRNN", self)
        mini_batch = mini_batch.to(self.device)
        mini_batch_reversed = mini_batch_reversed.to(self.device)
        mini_batch_mask = mini_batch_mask.to(self.device)
        T_max = mini_batch.size(1)
        batch_size = mini_batch.size(0)
        dim_h = self.gen_RNN.hidden_size

        # p(z_0)
        z_prev = self.z_0.expand(batch_size, self.z_0.size(0))
        h_0_contig = self.h_0.expand(1, batch_size, dim_h).contiguous()
        packed_mini_batch = pack_padded_sequence(mini_batch, mini_batch_seq_lengths, batch_first=True, enforce_sorted=True)
        packed_h_output, _ = self.gen_RNN(packed_mini_batch, h_0_contig)
        h_output, _ = pad_packed_sequence(packed_h_output, batch_first=True)

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                # p(z_t|z_{t-1}, h_{t-1}^1)
                if t == 1:
                    dz_input = torch.cat((z_prev, h_0_contig[0,:,:]), dim=-1)
                else:
                    dz_input = torch.cat((z_prev, h_output[:,t-2,:]), dim=-1)
                z_loc, z_scale = self.dz(dz_input)
                with poutine.scale(scale=annealing_factor):
                    z_prev = pyro.sample("z_%d" % t, dist.Normal(z_loc, z_scale).mask(mini_batch_mask[:, t - 1 : t]).to_event(1))
                # p(x_t|z_t, h_{t-1})
                if t == 1:
                    dx_input = torch.cat((z_prev, h_0_contig[0,:,:]), dim=-1)
                else:
                    dx_input = torch.cat((z_prev, h_output[:,t-2,:]), dim=-1)
                x_ps = self.dx(dx_input)
                pyro.sample("obs_x_%d" % t, dist.Bernoulli(probs=x_ps).mask(mini_batch_mask[:, t - 1 : t]).to_event(1), obs=mini_batch[:, t - 1, :])

    def guide_1(self, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, annealing_factor=1.0):
        pyro.module("SRNN", self)
        mini_batch = mini_batch.to(self.device)
        mini_batch_reversed = mini_batch_reversed.to(self.device)
        mini_batch_mask = mini_batch_mask.to(self.device)
        T_max = mini_batch.size(1)
        batch_size = mini_batch.size(0)
        H = self.inf_RNN_bi.hidden_size # hidden size, dim_g

        g_0_contig = self.g_0_bi.expand(2, batch_size, H).contiguous()
        packed_mini_batch = pack_padded_sequence(mini_batch, mini_batch_seq_lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_output, _ = self.inf_RNN_bi(packed_mini_batch, g_0_contig)
        rnn_output, _ = pad_packed_sequence(packed_rnn_output, batch_first=True)

        # q(z_0)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc, z_scale = self.combiner_1(z_prev, rnn_output[:, t - 1, :H], rnn_output[:, t - 1, H:])
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                with poutine.scale(scale=annealing_factor):
                    if len(self.iafs) > 0:
                        z_t = pyro.sample("z_%d" % t, z_dist.mask(mini_batch_mask[:, t - 1]))
                    else:
                        z_t = pyro.sample("z_%d" % t, z_dist.mask(mini_batch_mask[:, t - 1:t]).to_event(1))
                # update time step
                z_prev = z_t

    def guide_2(self, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, annealing_factor=1.0):
        pyro.module("SRNN", self)
        mini_batch = mini_batch.to(self.device)
        mini_batch_reversed = mini_batch_reversed.to(self.device)
        mini_batch_mask = mini_batch_mask.to(self.device)
        T_max = mini_batch.size(1)
        batch_size = mini_batch.size(0)
        dim_h = self.gen_RNN.hidden_size
        dim_g = self.inf_RNN_bw.hidden_size

        g_0_contig = self.g_0_bw.expand(1, batch_size, dim_g).contiguous()
        g_output, _ = self.inf_RNN_bw(mini_batch_reversed, g_0_contig)
        g_output = poly.pad_and_reverse(g_output, mini_batch_seq_lengths)

        h_0_contig = self.h_0.expand(1, batch_size, dim_h).contiguous()
        packed_mini_batch = pack_padded_sequence(mini_batch, mini_batch_seq_lengths, batch_first=True, enforce_sorted=True)
        packed_h_output, _ = self.gen_RNN(packed_mini_batch, h_0_contig)
        h_output, _ = pad_packed_sequence(packed_h_output, batch_first=True)

        # q(z_0)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc, z_scale = self.combiner_3(z_prev, h_output[:, t-1, :], g_output[:, t - 1, :])
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                with poutine.scale(scale=annealing_factor):
                    if len(self.iafs) > 0:
                        z_t = pyro.sample("z_%d" % t, z_dist.mask(mini_batch_mask[:, t - 1]))
                    else:
                        z_t = pyro.sample("z_%d" % t, z_dist.mask(mini_batch_mask[:, t - 1:t]).to_event(1))
                # update time step
                z_prev = z_t
