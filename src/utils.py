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