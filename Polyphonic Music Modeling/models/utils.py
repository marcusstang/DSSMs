import torch
import torch.nn as nn

class emitter(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        h1 = self.relu(self.lin_z_to_hidden(inputs))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = self.sigmoid(self.lin_hidden_to_output(h2))
        return ps

class stoch_trans(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_loc = nn.Linear(hidden_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
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

# p(x_t | z_t)
class Emitter(nn.Module):
    def __init__(self, output_dim, z_dim, emission_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_loc = nn.Linear(emission_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(emission_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_t):
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        x_loc = self.lin_hidden_to_loc(h2)
        x_scale = torch.exp(self.lin_hidden_to_scale(h2))
        return x_loc, x_scale

# p(z_t | z_{t-1})
class GatedTransition(nn.Module):
    def __init__(self, z_dim, transition_dim):
        super().__init__()
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        return loc, scale