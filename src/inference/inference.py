import json
import torch
from sklearn.metrics import accuracy_score
from src.utils.load_dataset import get_data
from transformers import pipeline

def load_labels(jsonl_path):
    frame_to_label = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            raw_frame = entry["frames_file_names"][0]
            label = entry["label"]
            frame_to_label[raw_frame] = label
    return frame_to_label

def predict(model_id, dataset, labels_map, max_examples=10):
    labels = []
    preds = []
    examples = dataset[:max_examples]
    pipe = pipeline("image-text-to-text", model=model_id)

    for example in examples:
        example_path = example[1]["content"][0]["image"]
        label = labels_map.get(example_path)
        labels.append(label)
        decoded = pipe(text=example)
        pred = "no" if "no, remove the element" in decoded else "yes"
        preds.append(pred)

    return labels, preds

def main():
    model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    train_path = "data/train-visibility.jsonl"
    test_path = "data/test-visibility.jsonl"
    task = "visibility"

    train_dataset, test_dataset = get_data(task, train_path, test_path, add_label=False)
    labels_map = load_labels(test_path)
    labels, preds = predict(model_id, test_dataset, labels_map, max_examples=50)
    accuracy = accuracy_score(labels, preds)
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
 