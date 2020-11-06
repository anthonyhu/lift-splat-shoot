import numpy as np
import torch
import cv2
from PIL import Image

def print_model_spec(model, name=''):
    n_parameters = count_n_parameters(model)
    n_trainable_parameters = count_n_parameters(model, only_trainable=True)
    print(f'Model {name}: {n_parameters:.2f}M parameters of which {n_trainable_parameters:.2f}M are trainable.')


def count_n_parameters(model, only_trainable=False):
    if only_trainable:
        n_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    else:
        n_parameters = sum([p.numel() for p in model.parameters()])
    return n_parameters / 10**6


def set_module_grad(module, requires_grad=False):
    for p in module.parameters():
        p.requires_grad = requires_grad


def convert_figure_numpy(figure):
    """ Convert figure to numpy image """
    figure_np = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    figure_np = figure_np.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    return figure_np

def _generate_instance_colours(instance_map):
    return {
        instance_id: np.random.random(3)
        for instance_id in instance_map.values()
    }

def plot_instance_map(instance_image, instance_map, instance_colours={}):
    if isinstance(instance_image, torch.Tensor):
        instance_image = instance_image.numpy()
    assert isinstance(instance_image, np.ndarray)
    if not instance_colours:
        instance_colours = _generate_instance_colours(instance_map)
    plot_image = np.zeros((instance_image.shape[0], instance_image.shape[1], 3))
    for key, value in instance_colours.items():
        plot_image[instance_image==key] = value

    return plot_image

def np_uint8_to_pil(np_img: np.ndarray) -> Image.Image:
    if np_img.dtype != np.uint8:
        raise TypeError(f"Expected np.ndarray of dtype np.uint8, but got dtype {np_img.dtype}")
    if np_img.ndim == 3 and np_img.shape[-1] == 1:
        np_img = np.squeeze(np_img)
    elif np_img.ndim != 2:
        raise ValueError(f"Unsupported shape {np_img.shape}")

    pil_img = Image.fromarray(np_img, mode='L')
    return pil_img
    
def flow_to_image(flow: np.ndarray, autoscale: bool = True) -> np.ndarray:
    """
    Applies colour map to flow which should be a 2 channel image tensor HxWx2. Returns a HxWx3 numpy image
    Code adapted from: https://github.com/liruoteng/FlowNet/blob/master/models/flownet/scripts/flowlib.py
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Convert to polar coordinates
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = np.max(rad)

    # Normalise flow maps
    if autoscale:
        u /= (maxrad + np.finfo(float).eps)
        v /= (maxrad + np.finfo(float).eps)

    # visualise flow with cmap
    return compute_color(u, v)


def compute_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert u.shape == v.shape
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nan_mask = np.isnan(u) | np.isnan(v)
    u[nan_mask] = 0
    v[nan_mask] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    f_k = (a + 1) / 2 * (ncols - 1) + 1
    k_0 = np.floor(f_k).astype(int)
    k_1 = k_0 + 1
    k_1[k_1 == ncols + 1] = 1
    f = f_k - k_0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k_0 - 1] / 255
        col1 = tmp[k_1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = col * (1 - nan_mask)

    return img

def make_color_wheel() -> np.ndarray:
    """
    Create colour wheel.
    Code adapted from https://github.com/liruoteng/FlowNet/blob/master/models/flownet/scripts/flowlib.py
    """
    red_yellow = 15
    yellow_green = 6
    green_cyan = 4
    cyan_blue = 11
    blue_magenta = 13
    magenta_red = 6

    ncols = red_yellow + yellow_green + green_cyan + cyan_blue + blue_magenta + magenta_red
    colorwheel = np.zeros([ncols, 3])

    col = 0

    # red_yellow
    colorwheel[0:red_yellow, 0] = 255
    colorwheel[0:red_yellow, 1] = np.transpose(np.floor(255 * np.arange(0, red_yellow) / red_yellow))
    col += red_yellow

    # yellow_green
    colorwheel[col:col + yellow_green, 0] = 255 - np.transpose(
        np.floor(255 * np.arange(0, yellow_green) / yellow_green))
    colorwheel[col:col + yellow_green, 1] = 255
    col += yellow_green

    # green_cyan
    colorwheel[col:col + green_cyan, 1] = 255
    colorwheel[col:col + green_cyan, 2] = np.transpose(np.floor(255 * np.arange(0, green_cyan) / green_cyan))
    col += green_cyan

    # cyan_blue
    colorwheel[col:col + cyan_blue, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, cyan_blue) / cyan_blue))
    colorwheel[col:col + cyan_blue, 2] = 255
    col += cyan_blue

    # blue_magenta
    colorwheel[col:col + blue_magenta, 2] = 255
    colorwheel[col:col + blue_magenta, 0] = np.transpose(np.floor(255 * np.arange(0, blue_magenta) / blue_magenta))
    col += + blue_magenta

    # magenta_red
    colorwheel[col:col + magenta_red, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, magenta_red) / magenta_red))
    colorwheel[col:col + magenta_red, 0] = 255

    return colorwheel