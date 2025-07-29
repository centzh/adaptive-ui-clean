import json
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.utils.load_dataset import get_data
from src.utils.seed import set_seed
from transformers import pipeline
import argparse
import ipdb
import random

set_seed()

def load_labels(jsonl_path):
    """
    Loads frame-to-label mappings from a JSONL file.

    Parameters
    ----------
    jsonl_path : str
        Path to the JSONL file containing frame file names and corresponding labels.

    Returns
    -------
    dict
        A dictionary mapping frame file names to their labels.
    """
    frame_to_label = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            raw_frame = entry["frames_file_names"][0]
            label = entry["label"]
            frame_to_label[raw_frame] = label
    return frame_to_label

def predict(model_id, dataset, labels_map, max_examples=50):
    """
    Performs zero-shot image-text-to-text inference on a dataset using a specified model.

    Parameters
    ----------
    model_id : str
        The identifier of the model to use for inference.
    dataset : list
        The dataset to evaluate, where each entry is a tuple containing metadata and input content.
    labels_map : dict
        A dictionary mapping frame file names to ground-truth labels.
    max_examples : int, optional
        The maximum number of examples to evaluate (default is 50).

    Returns
    -------
    tuple of list
        A tuple containing two lists:
        - Ground-truth labels
        - Model predictions ("yes" or "no")
    """
    labels = []
    preds = []
    examples = random.sample(dataset, k=max_examples)
    device = 0 if torch.cuda.is_available() else -1
    print(device)
    pipe = pipeline("image-text-to-text", model=model_id, device=device)
 
    for example in examples:

        example_path = example[1]["content"][0]["image"]
        label = labels_map.get(example_path)
        labels.append(label)

        if label is None:
            print(example_path)
            print(f"Warning: No label found for {example_path}, skipping...")
            continue
       
        decoded = pipe(text=example, max_new_tokens=512)
        response = decoded[0]['generated_text'][2]["content"].lower()
        pred = "no" if "no, remove the element" in response else "yes"
        preds.append(pred)

    return labels, preds
 

def parse_args():
    """
    Parses command-line arguments for the zero-shot inference script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including dataset paths, config path, output directory, model id, and run name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="visibility")
    parser.add_argument("--train_path", type=str, default="data/train-visibility.jsonl")
    parser.add_argument("--test_path", type=str, default="data/test-visibility.jsonl")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--question_type", type=str, default="simple")
    return parser.parse_args()


def main():
    args = parse_args()
    task = args.task
    train_path = args.train_path
    test_path = args.test_path
    model_id = args.model_id
    question_type = args.question_type

    train_dataset, test_dataset = get_data(task, train_path, test_path, question_type, add_label=False)
    labels_map = load_labels(train_path)
  
    labels, preds = predict(model_id, train_dataset, labels_map, max_examples=200)
    accuracy = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, pos_label='yes')
    precision = precision_score(labels, preds, pos_label='yes')

    model_id = model_id.split("/")[1]

    with open(f"results/zero-shot-{model_id}-{question_type}-prompt.txt", "w") as f:
        f.write("=== Evaluation Report ===\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")

if __name__ == "__main__":
    main()
 
