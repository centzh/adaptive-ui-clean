from datasets import load_dataset
from src.utils.questions import *

def load_data(train_path, test_path):
    data_files = {"train": train_path, "test": test_path}
    dataset = load_dataset("json", data_files=data_files)
    print("Successfully loaded dataset")
    return dataset 

def format_data(task, sample, question_type="structured", add_label=True):
    """
    Formats a data sample into the chat-based input-output structure for the model.

    Parameters
    ----------
    task: str
        Visibility or placement task.
    sample: dict
        Raw sample containing 'frames_file_names' and 'label'.
    add_label: bool
        Variable to indicate whether to add labels to the data (e.g. for training) or not (e.g. for inference).
    question_type: str
        Type of question (prompt): simple, structured (CoT), few shot (CoT with example).

    Returns
    -------
    formatted_sample: list
        List of roles and content dictionaries structured for training.
    """
    if task == "visibility":
        if question_type == "simple":
            question = question_visibility_simple
        elif question_type == "few_shot":
            question = question_visibility_structured_with_few_shot
        else:
            question = question_visibility_structured
        response = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                                {"type": "image", "image": sample["frames_file_names"][0]},
                                {"type": "image", "image": sample["frames_file_names"][1]},
                                {"type": "text", "text": question}
            ]}
        ]

        if add_label:
            response.append(  {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"]}],
            })

        return response
  
    elif task == "placement":
        response = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                                {"type": "image", "image": sample["frames_file_names"][0]},
                                {"type": "image", "image": sample["frames_file_names"][1]},
                                {"type": "image", "image": sample["frames_file_names"][2]},
                                {"type": "text", "text": question_placement_structured}
            ]},
        ]

        if add_label:
            response.append(  {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"]}],
            })
        return response

    else:
        raise ValueError("Incorrect task entered.")

def preprocess_data(task, train_dataset, test_dataset, question_type="structured", add_label=True):
    train_dataset = [format_data(task, sample, question_type, add_label) for sample in train_dataset]
    test_dataset = [format_data(task, sample, question_type, add_label) for sample in test_dataset]
    return train_dataset, test_dataset
    
def get_data(task, train_path, test_path, question_type="structured", add_label=True):
    """
    Loads and preprocesses training and test datasets.

    Parameters
    ----------
    task: str
        Visibility or placement task
    train_path: str
        File path to the training dataset.
    test_path: str
        File path to the test dataset.
    question_type: str
        Type of question (prompt): simple, structured.
    add_label: bool
        Variable to indicate whether to add labels to the data.

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
    train_dataset, test_dataset = preprocess_data(task, train_dataset, test_dataset, question_type, add_label)
    return train_dataset, test_dataset