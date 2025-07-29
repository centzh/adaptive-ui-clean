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
from src.utils.load_dataset import get_data
from src.utils.manage_gpu_memory import clear_memory
from src.utils.load_model import load_model, get_processor
 
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

def parse_args():
    """
    Parses command-line arguments for the training script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including dataset paths, config path, output directory, model id, and run name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="visibility")
    parser.add_argument("--train_path", type=str, default="data/train-visibility.jsonl")
    parser.add_argument("--test_path", type=str, default="data/test-visibility.jsonl")
    parser.add_argument("--config_path", type=str, default="src/training/training.yml")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    parser.add_argument("--run_name", type=str, default="default-run")
    parser.add_argument("--min_patches", type=int, default=4)
    parser.add_argument("--max_patches", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=8)
    return parser.parse_args()

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
    task = args.task
    train_path = args.train_path
    test_path = args.test_path
    config_path = args.config_path
    output_dir = args.output_dir
    model_id = args.model_id
    run_name = args.run_name
    min_patches = args.min_patches
    max_patches = args.max_patches
    lora_rank = args.lora_rank

    train_dataset, test_dataset = get_data(task, train_path, test_path)
    train(model_id, run_name, train_dataset, test_dataset, config_path, min_patches, max_patches, lora_rank)
    
if __name__ == "__main__":
    main()