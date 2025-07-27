from datasets import load_dataset
import ipdb
import gc
import time
# from transformers import BitsAndBytesConfig
import torch
from trl import SFTConfig, SFTTrainer
import wandb
from peft import LoraConfig, get_peft_model
import random
import numpy as np
import os
from transformers import EarlyStoppingCallback
os.environ["WANDB_MODE"] = "online"
print(torch.version.cuda)        # CUDA version PyTorch was built with

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

# from transformers import BitsAndBytesConfig
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

def get_peft_config():
    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    return peft_config

def get_trainer(model, training_args, train_dataset, eval_dataset, data_collator, peft_config, processing_class):
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

def get_training_args():
    # Configure training arguments
    training_args = SFTConfig(
        output_dir="qwen2-32b-instruct-trl-sft-ChartQA",  # Directory to save the model
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=1,  # Batch size for training
        per_device_eval_batch_size=1,  # Batch size for evaluation
        gradient_accumulation_steps=10,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=10,  # Steps interval for logging
        eval_steps=100,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=100,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision

        # CHANGE BACK
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=1024  # Maximum sequence length for input
    )
    training_args.remove_unused_columns = False  # Keep unused columns in dataset
    return training_args

def format_data(sample):
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
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    # if "bnb_config" in globals():
    #     del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    # if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
    #     image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    # else:
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch

# Load this from zero-shot.py later
def load_model(model_id: str):

    # BitsAndBytesConfig int-4 config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    # )
 
    print(f"Loading {model_id}")
    if "Qwen2.5-VL" in model_id:
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model identifier: {model_id}")
    
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # quantization_config=bnb_config
        # attn_implementation="flash_attention_2",
    )

    # GOT RID OF EVAL
  
    print(f"Successfully loaded {model_id} for inference")
    return model

def connect_w_and_b(training_args, run_name):
    wandb.init(
        project="adaptive-ui",  
        name=run_name,
        config=training_args,
    )

# MODIFIED MIN AND MAX PIXELS
def get_processor(model_id):
    processor = AutoProcessor.from_pretrained(
        model_id, 
        min_pixels = 256 * 28 * 28, 
        max_pixels = 512 * 28 * 28
    )
    return processor

if __name__ == "__main__":
    train_path = "data/train-task-2.jsonl"
    test_path = "data/test-task-2.jsonl"
    dataset = load_data(train_path, test_path)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    train_dataset, test_dataset = preprocess_data(train_dataset, test_dataset)
# 
    clear_memory()

    # model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    model = load_model(model_id)
    processor = get_processor(model_id)

    training_args = get_training_args()
    run_name = "task-2-qwen2.5-vl-32b-instruct-full-dataset"
    connect_w_and_b(training_args, run_name)
    peft_config = get_peft_config()
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
    # trainer.save_model(training_args.output_dir)
     
