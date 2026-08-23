from PIL import Image
import torch
from torchvision import transforms
import numpy as np


def extract(model: torch.nn.Module, image_path, processor=None, **kwargs):
    """
    Intelligently extracts an image segmentation mask from a given image using either a standard
    PyTorch model or a Processor-based foundation model.

    This function takes an image and processes it based on the model strategy. If a processor is supplied,
    it delegates preprocessing (e.g., resizing, scaling, prompt-handling) to the processor. Otherwise,
    it applies standard manual transformations based on provided kwargs. It then feeds the input into
    the model to generate a segmentation mask. The original image and the mask are returned as numpy arrays.

    Args:
        model (torch.nn.Module): The pretrained PyTorch model to use for segmentation predictions.
        image_path (str): The path to the image file on which to perform segmentation.
        processor (Any, optional): The model-specific processor (e.g., from Hugging Face) used for preprocessing
                                   inputs. If None, standard PyTorch manual transformations are applied.
                                   Default is None.
        **kwargs: Additional arbitrary keyword arguments for model-specific configurations.
                  For standard models (e.g., BiRefNet): `resize_dim`, `normalize_mean`, `normalize_std`, `output_index`, `output_key`.
                  For processor models (e.g., SAM): Specific prompt arguments like `input_points`.

    Raises:
        FileNotFoundError: If the image file at `image_path` does not exist.
        ValueError: If the model is incompatible with the task or the processor output format is unrecognized.

    Returns:
        tuple (numpy.ndarray, numpy.ndarray): The original image converted to a numpy array,
                                              and the extracted segmentation mask as a numpy array.
    """
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    elif isinstance(image_path, np.ndarray):
        image = Image.fromarray(image_path.astype(np.uint8)).convert("RGB")
    else:
        raise TypeError("image_path must be a file path or an RGB NumPy array")

    # Processor-Based Models (SAM)
    if processor is not None:
        # Pass the image and any SAM-specific kwargs (like input_points) to the processor
        inputs = processor(image, return_tensors="pt", **kwargs).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Handle SAM-specific output format
        if hasattr(outputs, "pred_masks"):
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            # Extract the best mask for the first point
            best_mask = masks[0][0][0].numpy()
            mask_np = (best_mask * 255).astype(np.uint8)

        else:
            raise ValueError("Unrecognized processor output format.")

        del inputs, outputs, masks

    # Standard Models (BiRefNet)
    else:
        # Extract BiRefNet-specific kwargs with safe defaults
        resize_dim = kwargs.get("resize_dim", (1024, 1024))
        normalize_mean = kwargs.get("normalize_mean", [0.485, 0.456, 0.406])
        normalize_std = kwargs.get("normalize_std", [0.229, 0.224, 0.225])

        transform = transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(input_tensor)

            # BiRefNet returns a tuple/list of tensors; we want the last one
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]

            preds = preds.sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask_np = np.array(mask)

        del input_tensor, preds

    # Clean up and Return
    image_np = np.array(image)
    torch.cuda.empty_cache()

    return image_np, mask_np
