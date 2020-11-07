import torch


def probabilistic_kl_loss(output):
    present_mu = output['present_mu']
    present_log_sigma = output['present_log_sigma']
    future_mu = output['future_mu']
    future_log_sigma = output['future_log_sigma']

    var_future = torch.exp(future_log_sigma) ** 2
    var_present = torch.exp(present_log_sigma) ** 2
    kl_div = (
            present_log_sigma
            - future_log_sigma
            - 0.5
            + (var_future + (future_mu - present_mu) ** 2) / (2 * var_present)
    )

    kl_loss = torch.mean(torch.sum(kl_div, dim=-1))

    return kl_loss
