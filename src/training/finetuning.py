from datasets import load_dataset
import gc
import time
import torch
import ipdb
from trl import SFTConfig, SFTTrainer
import wandb
from peft import LoraConfig, get_peft_model
import random
import numpy as np
import os
import yaml
import argparse
from transformers import EarlyStoppingCallback
os.environ["WANDB_MODE"] = "online"
 
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

question = (
    "You are shown two identical images from a head‑mounted camera:\n"
    "- Image 1 is the original camera view.\n"
    "- Image 2 is the same view but with a UI element overlayed on part of the scene.\n"
    "\n"
    "Step 1: What does the UI element cover?\n"
    "'Covers' means the element hides part of an object from view. If the object is fully visible around the element, it is not covered.\n"
    "Answer format: The element covers [object/area].\n"
    "\n"
    "Step 2: Answer each question with 'yes' or 'no':\n"
    "- Does the element cover something the user is using, about to use, or interacting with?\n"
    "- Does the element cover the floor, doors, windows or signs?\n"
    "- Does the element cover people's faces or bodies?\n"
    "- Does the element cover an area of high colour intensity or edge contrast?\n"
    "IMPORTANT RULE:\n"
    "- If **any** answer in Step 2 is 'yes' → FINAL ANSWER: No, remove the element.\n"
    "- If **all** answers in Step 2 are 'no' → FINAL ANSWER: Yes, keep the element.\n"
    "\n"
    "STRICT FORMAT:\n"
    "Step 1: [Your answer here]\n"
    "Step 2:\n"
    "- Covers floor/doors/windows/signs: [yes/no]\n"
    "- Covers people: [yes/no]\n"
    "- Covers high‑contrast or high‑colour‑intensity area: [yes/no]\n"
    "FINAL ANSWER: [Yes, keep the element | No, remove the element]"
)

def get_peft_config(lora_rank, lora_alpha=16, lora_dropout=0.05):
    """
    Returns a LoRA PEFT configuration object for fine-tuning.

    Parameters
    ----------
    lora_rank: int, optional
        Rank parameter for LoRA layers (default is 8).
    lora_alpha: int, optional
        Alpha scaling factor for LoRA layers (default is 16, in practice).
    lora_dropout: float, optional
        Probability of dropout for LoRA layers.

    Returns
    -------
    peft_config: LoraConfig
        Configured LoRA PEFT settings for causal language modeling.
    """
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    return peft_config

def get_trainer(model, training_args, train_dataset, eval_dataset, data_collator, peft_config, processing_class):
    """
    Returns an SFTTrainer configured for training with early stopping.

    Parameters
    ----------
    model: torch.nn.Module
        The model to be fine-tuned.
    training_args: SFTConfig
        Training arguments and configurations.
    train_dataset: Dataset
        Dataset used for training.
    eval_dataset: Dataset
        Dataset used for evaluation.
    data_collator: callable
        Function to collate batches during training.
    peft_config: LoraConfig
        LoRA PEFT configuration.
    processing_class: tokeniser or processor
        Class responsible for processing input data.

    Returns
    -------
    trainer: SFTTrainer
        Initialised trainer ready for training.
    """
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        peft_config=peft_config,
        processing_class=processing_class,
        # FIX THIS
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]

    )
    return trainer

def get_training_args(config_path: str):
    """
    Loads training arguments from a YAML configuration file.

    Parameters
    ----------
    config_path: str
        Path to the YAML config file.

    Returns
    -------
    training_args: SFTConfig
        Training configuration object populated from the YAML file.
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    print(config_dict)
    training_args = SFTConfig(**config_dict)
    return training_args


def format_data(sample):
    """
    Formats a data sample into the chat-based input-output structure for the model.

    Parameters
    ----------
    sample: dict
        Raw sample containing 'frames_file_names' and 'label'.

    Returns
    -------
    formatted_sample: list
        List of roles and content dictionaries structured for training.
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
                            {"type": "image", "image": sample["frames_file_names"][0]},
                            {"type": "image", "image": sample["frames_file_names"][1]},
                            {"type": "text", "text": question}
        ]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"]}],
        },
    ]
 
def load_data(train_path, test_path):
    data_files = {"train": train_path, "test": test_path}
    dataset = load_dataset("json", data_files=data_files)
    print("Successfully loaded dataset")
    return dataset 

def preprocess_data(train_dataset, test_dataset):
    train_dataset = [format_data(sample) for sample in train_dataset]
    test_dataset = [format_data(sample) for sample in test_dataset]
    return train_dataset, test_dataset

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

def create_collate_fn(processor):
    """
    Creates a collate function with access to the processor.
    
    Parameters
    ----------
    processor: AutoProcessor
        The processor to use for tokenization and image processing.
        
    Returns
    -------
    collate_fn: callable
        Collate function that can be used with DataLoader.
    """
    def collate_fn(examples):
        """
        Collates a batch of examples by processing images and text inputs into tensors.

        Parameters
        ----------
        examples: list
            List of formatted data examples.

        Returns
        -------
        batch: dict
            Dictionary containing input tensors and masked labels ready for training.
        """
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example, tokenize=False) for example in examples
        ]  
        
        # Prepare texts for processing
        image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        ) 

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        # Add labels to the batch
        batch["labels"] = labels  
        return batch
    
    return collate_fn

# Load this from zero-shot.py later
def load_model(model_id: str):

    print(f"Loading {model_id}")
    if "Qwen2.5-VL" in model_id:
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model identifier: {model_id}")
    
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Successfully loaded {model_id} for inference")
    return model

def connect_w_and_b(training_args, run_name):
    """
    Initialises a Weights & Biases run with the given configuration.

    Parameters
    ----------
    training_args: SFTConfig
        Training configuration to log.
    run_name: str
        Name of the wandb run.
    """
    wandb.init(
        project="adaptive-ui-clean",  
        name=run_name,
        config=training_args,
    )

def get_processor(model_id, min_patches, max_patches, patch_size=28):
    """
    Loads and returns the processor for the given model with patch size settings.

    Parameters
    ----------
    model_id: str
        Identifier of the model for which to load the processor.
    min_patches: int, optional
        Minimum number of patches to process.
    max_patches: int, optional
        Maximum number of patches to process.
    patch_size: int, optional
        Size of each patch.

    Returns
    -------
    processor: AutoProcessor
        Processor configured for the model.
    """
    processor = AutoProcessor.from_pretrained(
        model_id, 
        min_pixels = min_patches * patch_size * patch_size, 
        max_pixels = max_patches * patch_size * patch_size
    )
    return processor

def parse_args():
    """
    Parses command-line arguments for the training script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including dataset paths, config path, output directory, model id, and run name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train-task-2.jsonl")
    parser.add_argument("--test_path", type=str, default="data/test-task-2.jsonl")
    parser.add_argument("--config_path", type=str, default="src/training/training.yml")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--run_name", type=str, default="default-run")
    parser.add_argument("--min_patches", type=int, default=256)
    parser.add_argument("--max_patches", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=8)
    return parser.parse_args()

def get_data(train_path, test_path):
    """
    Loads and preprocesses training and test datasets.

    Parameters
    ----------
    train_path: str
        File path to the training dataset.
    test_path: str
        File path to the test dataset.

    Returns
    -------
    train_dataset: list
        Preprocessed training dataset.
    test_dataset: list
        Preprocessed test dataset.
    """
    dataset = load_data(train_path, test_path)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_dataset, test_dataset = preprocess_data(train_dataset, test_dataset)
    return train_dataset, test_dataset


def train(model_id, run_name, train_dataset, test_dataset, config_path, min_patches, max_patches, lora_rank):
    """
    Runs the full training pipeline: clear memory, load model and processor, prepare training, and start training.

    Parameters
    ----------
    model_id: str
        Identifier for the pretrained model to fine-tune.
    run_name: str
        Name of the current training run for logging.
    train_dataset: list
        Preprocessed training dataset.
    test_dataset: list
        Preprocessed test dataset.
    config_path: str
        Path to the YAML config for training.
    min_patches: int
        Mininmum number of patches to divide the input image.
    max_patches: int
        Maximum number of patches to divide the input image.
    lora_rank : int
        LoRA rank parameter.
    """

    # Clear GPU memory
    clear_memory()

    # Load memory
    model = load_model(model_id)
    processor = get_processor(model_id, min_patches, max_patches)

    training_args = get_training_args(config_path)
    connect_w_and_b(training_args, run_name)
    peft_config = get_peft_config(lora_rank)

    collate_fn = create_collate_fn(processor)

    trainer = get_trainer(
        model=model, 
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


def main():
    """Main training pipeline"""
    args = parse_args()
    train_path = args.train_path
    test_path = args.test_path
    config_path = args.config_path
    output_dir = args.output_dir
    model_id = args.model_id
    run_name = args.run_name
    min_patches = args.min_patches
    max_patches = args.max_patches
    lora_rank = args.lora_rank

    train_dataset, test_dataset = get_data(train_path, test_path)
    train(model_id, run_name, train_dataset, test_dataset, config_path, min_patches, max_patches, lora_rank)
    
if __name__ == "__main__":
    main()