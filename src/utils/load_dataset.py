from datasets import load_dataset
from src.utils.questions import question_placement, question_visibility

def load_data(train_path, test_path):
    data_files = {"train": train_path, "test": test_path}
    dataset = load_dataset("json", data_files=data_files)
    print("Successfully loaded dataset")
    return dataset 

def format_data(task, sample, add_label=True):
    """
    Formats a data sample into the chat-based input-output structure for the model.

    Parameters
    ----------
    task: str
        Visibility or placement task
    sample: dict
        Raw sample containing 'frames_file_names' and 'label'.
    add_label: bool
        Variable to indicate whether to add labels to the data (e.g. for training) or not (e.g. for inference)

    Returns
    -------
    formatted_sample: list
        List of roles and content dictionaries structured for training.
    """
    if task == "visibility":
        response = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                                {"type": "image", "image": sample["frames_file_names"][0]},
                                {"type": "image", "image": sample["frames_file_names"][1]},
                                {"type": "text", "text": question_visibility}
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
                                {"type": "text", "text": question_placement}
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

def preprocess_data(task, train_dataset, test_dataset, add_label=True):
    train_dataset = [format_data(task, sample, add_label) for sample in train_dataset]
    test_dataset = [format_data(task, sample, add_label) for sample in test_dataset]
    return train_dataset, test_dataset
    
def get_data(task, train_path, test_path, add_label=True):
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
    add_label: bool
        Variable to indicate whether to add labels to the data

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
    train_dataset, test_dataset = preprocess_data(task, train_dataset, test_dataset, add_label)
    return train_dataset, test_dataset