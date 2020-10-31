import torch
import torch.nn as nn

from common.constants import MEAN, STD


class ImagePreprocessing(nn.Module):
    def __init__(self):
        super().__init__()
        # Automatically broadcast to batch and time dimensions
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

        self.mean = torch.nn.Parameter(self.mean, requires_grad=False)
        self.std = torch.nn.Parameter(self.std, requires_grad=False)

    def forward(self, x):
        x = x.sub_(self.mean).div_(self.std)
        return x


def one_hot(x, n_categories):
    if x.ndim == 1:
        final_shape = (x.shape[0], n_categories)
    else:
        final_shape = x.shape[:-1] + (n_categories,)

    x = x.view(-1, 1)
    one_hot_x = torch.zeros((x.shape[0], n_categories), dtype=torch.float32)
    one_hot_x.scatter_(1, x, 1)

    one_hot_x = one_hot_x.view(final_shape)
    return one_hot_x
