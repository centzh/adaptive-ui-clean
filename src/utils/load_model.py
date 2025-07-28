import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    LlavaForConditionalGeneration
)

def load_model(model_id: str):

    print(f"Loading {model_id}")
    if "Qwen2.5-VL" in model_id:
        model_class = Qwen2_5_VLForConditionalGeneration
    elif "gemma-3" in model_id:
        model_class = Gemma3ForConditionalGeneration
    elif "llava" in model_id:
        model_class = LlavaForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model identifier: {model_id}")
    
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Successfully loaded {model_id} for inference")
    return model


def get_processor(model_id, min_patches=None, max_patches=None, patch_size=28):
    """
    Loads and returns the processor for the given model with patch size settings.

    Parameters
    ----------
    model_id: str
        Identifier of the model for which to load the processor.
    min_patches: int, optional
        Minimum number of patches to process (used in Qwen models only).
    max_patches: int, optional
        Maximum number of patches to process (used in Qwen models only).
    patch_size: int, optional
        Size of each patch (used in Qwen models only).

    Returns
    -------
    processor: AutoProcessor
        Processor configured for the model.
    """
    if "Qwen2.5-VL" in model_id:
        processor = AutoProcessor.from_pretrained(
            model_id, 
            min_pixels = min_patches * patch_size * patch_size, 
            max_pixels = max_patches * patch_size * patch_size
        )
    elif "gemma-3" in model_id or "llava" in model_id:
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        raise ValueError(f"Unsupported model identifier: {model_id}")

    return processor