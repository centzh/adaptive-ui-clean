import os
import json
import random
import numpy as np
import ipdb
from sklearn.model_selection import train_test_split
from PIL import Image
from detectors import SaliencyDetector
from instance_gen import InstanceGenerator
from seed import set_seed
from pathlib import Path

class DatasetGenerator:
    def __init__(
        self,
        processed_path,
        video_path,
        detector,
        task=2,
        element_size=400,
        step_size=20,
        seed=42,
        test_size=0.3
    ):
        self.processed_path = processed_path
        self.video_path = video_path
        self.task = task
        self.seed = seed
        self.test_size = test_size
        
        set_seed(seed)
        self.instance_generator = InstanceGenerator(detector, element_size, step_size)
        
        self.videos = self._get_videos()
        self.train_videos, self.test_videos = self._split_videos()
        
    def _get_videos(self):
        videos = [
            d for d in os.listdir(self.video_path) 
            if os.path.isdir(os.path.join(self.video_path, d)) 
            and os.listdir(os.path.join(self.video_path, d))
        ]
        return videos
    
    def _split_videos(self):
        train, test = train_test_split(self.videos, test_size=self.test_size, random_state=self.seed)
        return train, test
    
    def generate_dataset(self, split="train", eye_gaze_data=None, save_metadata_path=None):
        if split == "train":
            videos = self.train_videos
        elif split == "test":
            videos = self.test_videos
        else:
            raise ValueError(f"Invalid split '{split}'. Use 'train' or 'test'.")
        
        dataset = []
        
        for video in videos:
            frame_videos_path = os.path.join(self.video_path, video)
            
            for frame_id in os.listdir(frame_videos_path):
                frame_path = os.path.join(frame_videos_path, frame_id)
                frame = Image.open(frame_path)
                ipdb.set_trace()
                
                # Use instance generator to generate overlayed image and get label
                if self.task == 1 or self.task == 2:
                    generated_instance_data = self.instance_generator.generate(frame, frame_path, eye_gaze_data, self.task)
                    ipdb.set_trace()
                    dataset.append({
                        "frames_file_names": [frame_path, generated_instance_data["save_path"]],
                        "label": generated_instance_data["label"]
                    })
                    ipdb.set_trace()
                else:
                    pass
        
        if save_metadata_path:
            with open(save_metadata_path, "w") as f:
                for item in dataset:
                    f.write(json.dumps(item) + "\n")
            print(f"Saved dataset metadata to {save_metadata_path}")
        
        return dataset

if __name__ == "__main__":
    
    processed_path = Path("data") / "generated_overlays"
    video_path = Path("data") / "video_frames"

    task_id = 2
    detector = SaliencyDetector()
    dataset_gen = DatasetGenerator(processed_path, video_path, detector, task=task_id, seed=42)
    
    # Generate train dataset metadata and files
    dataset_gen.generate_dataset(split="train", eye_gaze_data=None, save_metadata_path="data/train-task-{task_id}.jsonl")
    
    # # Generate test dataset metadata and files
    # dataset_gen.generate_dataset(split="test", eye_gaze_data=None, save_metadata_path="data/test-task-{task_id}.jsonl")
