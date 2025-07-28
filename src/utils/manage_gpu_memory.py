import torch
import gc

def clear_memory():
    """
    Clears GPU memory and deletes specified global variables if they exist.
    """
    var_names = ["inputs", "model", "processor", "trainer", "peft_model"]
    # Delete variables from the global scope
    for var in var_names:
        if var in globals():
            del globals()[var]

    # Garbage collection and clearing CUDA memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("CUDA not available - using CPU")