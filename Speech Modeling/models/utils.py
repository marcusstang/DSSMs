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

class stoch_trans(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_loc = nn.Linear(hidden_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        h1 = self.relu(self.lin_z_to_hidden(inputs))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        output_loc = self.lin_hidden_to_loc(h2)
        output_scale = torch.exp(self.lin_hidden_to_scale(h2))
        return output_loc, output_scale

# q(z_t|z_{t-1}, g_t^l, g_t^r)
class Combiner_1(nn.Module):
    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t, g_fw_t, g_bw_t):
        h_combined = 1/3 * (self.tanh(self.lin_z_to_hidden(z_t)) + g_fw_t + g_bw_t)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return loc, scale

# q(z_t|h_{t-1}, g_{t-1}^l)
# we first use one fully connected layer to make h_{t-1} conformable with g_{t-1}^l
class Combiner_2(nn.Module):
    def __init__(self, z_dim, h_dim, g_dim):
        super().__init__()
        self.lin_h_to_g = nn.Linear(h_dim, g_dim)
        self.lin_hidden_to_loc = nn.Linear(g_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(g_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, htm1, gtm1):
        h_combined = 1/2 * (self.tanh(self.lin_h_to_g(htm1)) + gtm1)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return loc, scale

# q(z_t|h_{t-1}^1, h_t^2, g_{t}^l)
class Combiner_3(nn.Module):
    def __init__(self, z_dim, h1_dim, h2_dim, g_dim):
        super().__init__()
        self.lin_h1_to_g = nn.Linear(h1_dim, g_dim)
        self.lin_h2_to_g = nn.Linear(h2_dim, g_dim)
        self.lin_hidden_to_loc = nn.Linear(g_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(g_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, h1, h2, gt):
        h_combined = 1/3 * (self.tanh(self.lin_h1_to_g(h1)) + self.tanh(self.lin_h2_to_g(h2)) + gt)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return loc, scale

# q(z_t|z_{t-1}, g_t^l)
class Combiner_4(nn.Module):
    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t, g_bw_t):
        h_combined = 1/2 * (self.tanh(self.lin_z_to_hidden(z_t)) + g_bw_t)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return loc, scale
    
# Training one epoch of the training set
def train(svi, train_loader):
    epoch_loss = 0.
    for x, y in train_loader: # x is mini-batch
        epoch_loss += svi.step(x)

    # return average epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def train_KL_annealing(svi, train_loader, epoch, annealing_epochs, minimum_annealing_factor):
    batch_size = train_loader.batch_size
    N_mini_batches = len(train_loader)
    epoch_nll = 0.0
    for which_mini_batch, (x, _) in enumerate(train_loader):
        if annealing_epochs > 0 and epoch < annealing_epochs:
            annealing_factor = minimum_annealing_factor + (1.0 - minimum_annealing_factor) * (
                float(which_mini_batch + epoch * N_mini_batches + 1)
                / float(annealing_epochs * N_mini_batches)
            )
        else:
            annealing_factor = 1.0
        epoch_nll += svi.step(x, annealing_factor)
    # return average epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_nll / normalizer_train
    return total_epoch_loss_train

def mean_iaf(z_loc, z_scale, iaf_module, num_samples = 1000):
    base_dist = dist.Normal(z_loc, z_scale).to_event(1)
    transformed_dist = TransformedDistribution(base_dist, iaf_module)
    samples = transformed_dist.sample([num_samples])
    estimated_mean = samples.mean(dim=0)
    return estimated_mean