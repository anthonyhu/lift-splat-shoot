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


def cost_map_loss(output, future_trajectory, templates):
    if len(future_trajectory) == 0:
        return output['bev'].new_zeros(1)

    future_trajectory = future_trajectory.to(output['cost_map'].device)

    template_trajectories = templates['trajectories']
    template_row_indices = templates['row_indices']
    template_col_indices = templates['col_indices']
    n_templates, n_future_points, _ = template_trajectories.shape

    # present cost map
    cost_map = output['cost_map'][:, 0, 0]  # (batch, 200, 200)
    batch_size, h, w = cost_map.shape
    # Label is the closest to the template trajectories.
    # (batch, 1000, 10, 2)

    euclidean_distance = torch.norm(
        (future_trajectory.unsqueeze(1) - template_trajectories.unsqueeze(0)).view(batch_size, n_templates, -1), dim=-1)
    label_index = torch.argmax(euclidean_distance, dim=-1)

    # offset by 100 to put in the center, and multiplication by 2 to account for bev resolution
    batch_logits = []
    for b in range(batch_size):
        logits_list = []
        for i in range(n_templates):
            logits = cost_map.new_zeros(1)
            for t in range(n_future_points):
                if (0 <= template_row_indices[i, t] < h) and (0 <= template_col_indices[i, t] < w):
                    logits += cost_map[b, template_row_indices[i, t], template_col_indices[i, t]]
            logits_list.append(logits)

        logits_list = torch.cat(logits_list, dim=0)
        batch_logits.append(logits_list)

    # (batch, 1000)
    batch_logits = torch.stack(batch_logits, dim=0)

    return torch.nn.functional.cross_entropy(batch_logits, label_index)
