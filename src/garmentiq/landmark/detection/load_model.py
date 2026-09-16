import torch
from typing import Callable, Type


def load_model(model_path: str, model_class: Type[torch.nn.Module], device=None):
    """
    Load a PyTorch model from a checkpoint and prepare it for inference.

    Loads weights on CPU, then moves the supplied model instance to the requested
    device. A single-device inference model does not need DataParallel.

    Args:
        model_path (str): Path to the saved model checkpoint (.pth or .pt file).
        model_class (Type[torch.nn.Module]): The class definition of the model to be instantiated.
                                           This must be a subclass of `torch.nn.Module`.

    Raises:
        RuntimeError: If the model checkpoint cannot be loaded.

    Returns:
        torch.nn.Module: The loaded and ready-to-use model.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return model
