import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255, use_top_k=False, top_k_ratio=0.5):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio

    def forward(self, prediction, target):
        # shape (b*s, h, w)
        loss = nn.functional.cross_entropy(
            prediction, target, weight=self.weight, ignore_index=self.ignore_index, reduction='none'
        )
        # shape(b*s, h*w)
        loss = loss.view(loss.shape[0], -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[1])
            loss, _ = torch.sort(loss, dim=1, descending=True)
            loss = loss[:, :k]

        return torch.mean(loss)


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
    if len(output['cost_map'].shape) == 5:
        cost_map = output['cost_map'][:, 0, 0]  # (batch, 200, 200)
    else:
        cost_map = output['cost_map'][:, 0]
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

    # Mask invalid trajectories
    mask = future_trajectory.sum(dim=(-1, -2)) == 0

    if mask.sum() > 0:
        return torch.nn.functional.cross_entropy(batch_logits[mask], label_index[mask])
    return output['bev'].new_zeros(1)
