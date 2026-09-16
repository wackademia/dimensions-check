import inspect
import torch
import torch.nn as nn
from typing import Type
from safetensors.torch import load_file


def load_model(
    model_class: Type[nn.Module], model_path: str, model_args: dict = None, device=None, **kwargs
):
    """
    Loads a PyTorch model from a local checkpoint and prepares it for inference.

    This function instantiates the provided model class using safely filtered configuration
    arguments, loads the weights from a local `.pth` or `.safetensors` file, moves the model
    to the appropriate device (GPU or CPU), and sets it to evaluation mode. It automatically
    strips common weight prefixes (e.g., "module.", "model.") to ensure compatibility.

    Args:
        model_class (Type[nn.Module]): The uninstantiated PyTorch model class to be used.
        model_path (str): The local file path to the model checkpoint weights, typically
                          ending in `.pth` or `.safetensors`.
        model_args (dict, optional): A dictionary of configuration arguments for initializing
                                     the model. Incompatible arguments are safely ignored.
                                     Default is None.
        **kwargs: Additional arbitrary keyword arguments.

    Raises:
        Exception: If the model weights cannot be loaded from the specified local path or if
                   the file format is unsupported.

    Returns:
        nn.Module: The loaded and prepared PyTorch model instance.
    """
    model_args = model_args or {}
    
    # --- THE SMART CONVERTER ---
    # If the user passes a Config object instead of a dictionary, safely convert it.
    if not isinstance(model_args, dict):
        if hasattr(model_args, "to_dict"):
            model_args = model_args.to_dict()  # Hugging Face standard
        elif hasattr(model_args, "__dict__"):
            model_args = vars(model_args)      # Standard Python objects
        else:
            raise TypeError("model_args must be a dictionary or a configuration object.")
    # ---------------------------

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    sig = inspect.signature(model_class.__init__)
    valid_params = set(sig.parameters.keys())

    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    if not has_kwargs:
        filtered_args = {k: v for k, v in model_args.items() if k in valid_params}
    else:
        filtered_args = model_args

    # ... [Keep the rest of your loading logic exactly the same] ...
    model = model_class(**filtered_args)

    if model_path.endswith(".safetensors"):
        state_dict = load_file(model_path, device="cpu")
    else:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    new_state_dict = {
        k.removeprefix("module.").removeprefix("model."): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    return model
