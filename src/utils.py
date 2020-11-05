import numpy as np
import torch
import cv2

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
    