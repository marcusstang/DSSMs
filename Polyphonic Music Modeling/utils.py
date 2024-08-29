import torch
import pyro
from pyro.infer import Trace_ELBO
from pyro.infer.util import torch_item
from pyro.util import warn_if_nan

class TraceELBONLL(Trace_ELBO):
    def loss(self, model, guide, *args, **kwargs):
        nll = 0.0
        log_weights = []

        # Collect log importance weights for each particle
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            log_weight = torch_item(model_trace.log_prob_sum()) - torch_item(
                guide_trace.log_prob_sum()
            )
            log_weights.append(log_weight)

        # Convert log weights to tensor for stable log-sum-exp computation
        log_weights = torch.tensor(log_weights)

        # Compute the log of the mean of exponentiated weights (log-sum-exp trick)
        log_mean_weight = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(self.num_particles))

        loss = -log_mean_weight  # Negative log-likelihood
        warn_if_nan(loss, "loss")
        return loss

def get_nll(model, guide, *args, num_samples=100, **kwargs):
    log_weights = []

    for _ in range(num_samples):
        # Get the trace from the model and guide
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = pyro.poutine.trace(pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

        # Compute the log joint probability under the model
        log_joint = model_trace.log_prob_sum()

        # Compute the log joint probability under the guide
        log_guide = guide_trace.log_prob_sum()

        # Compute the log importance weight
        log_weight = log_joint - log_guide
        log_weights.append(log_weight)

    # Convert list of log weights to a tensor
    log_weights = torch.stack(log_weights)  # [num_samples, batch_size]

    # Compute log mean exp of log weights to estimate the marginal likelihood
    log_marginal_likelihood = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(num_samples, dtype=torch.float))

    # Normalize by the sequence lengths
    seq_lengths = kwargs.get('seq_lengths', None)
    if seq_lengths is not None:
        log_marginal_likelihood /= seq_lengths  # Normalize by each sequence length

    # Compute the negative log likelihood (NLL)
    nll = -log_marginal_likelihood.mean().item()

    return nll

def get_elbo(model, guide, *args, num_samples=1, **kwargs):
    log_weights = []

    for _ in range(num_samples):
        # Get the trace from the model and guide
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = pyro.poutine.trace(pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

        # Compute the log joint probability under the model
        log_joint = model_trace.log_prob_sum()

        # Compute the log joint probability under the guide
        log_guide = guide_trace.log_prob_sum()

        # Compute the log importance weight
        log_weight = log_joint - log_guide
        log_weights.append(log_weight)

    # Convert list of log weights to a tensor
    log_weights = torch.stack(log_weights).squeeze()  # [num_samples, batch_size]

    # Compute log mean exp of log weights to estimate the marginal likelihood
    elbo = torch.sum(log_weights, dim=0).item()


    return elbo


def get_elbo_1(svi, data_batch, data_batch_reversed, data_batch_mask, seq_lengths):
    # Compute the total ELBO by summing the ELBO for each sequence in the batch
    total_elbo = svi.evaluate_loss(data_batch, data_batch_reversed, data_batch_mask, seq_lengths)

    # Compute the total sequence length (sum of all T_i)
    total_seq_length = torch.sum(seq_lengths).item()

    # Compute the normalized ELBO loss
    normalized_loss = -total_elbo / total_seq_length

    return normalized_loss


def get_elbo_2(svi, data_batch, data_batch_reversed, data_batch_mask, seq_lengths):
    batch_size = len(seq_lengths)

    # Initialize a list to store normalized ELBOs for each sequence
    normalized_elbos = []

    # Iterate over each sequence in the batch
    for i in range(batch_size):
        # Extract the ith sequence data and corresponding reversed sequence, mask, and length
        seq_data = data_batch[:, i:i + 1]
        seq_data_reversed = data_batch_reversed[:, i:i + 1]
        seq_mask = data_batch_mask[:, i:i + 1]
        seq_length = seq_lengths[i]

        # Compute ELBO for this sequence
        elbo = svi.evaluate_loss(seq_data, seq_data_reversed, seq_mask, seq_length)

        # Normalize ELBO by the sequence length T_i
        normalized_elbo = elbo / seq_length

        # Store the normalized ELBO
        normalized_elbos.append(normalized_elbo)

    # Compute the final loss as the average of the normalized ELBOs
    loss = torch.mean(torch.tensor(normalized_elbos))

    return loss