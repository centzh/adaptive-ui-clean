"""
Dataset generation script for creating training and testing instances from video frame data.

This module loads video frames and eye-gaze data, then generates overlayed image instances
for a specified task using a saliency-based detector. The resulting datasets are saved
in JSON Lines format, containing frame file paths and corresponding labels.

Classes:
    DatasetGenerator: Handles dataset creation, video splitting, and instance generation.
"""
import json
import random
import numpy as np
import pandas as pd
import ipdb
from sklearn.model_selection import train_test_split
from PIL import Image
from detectors import SaliencyDetector
from instance_gen import InstanceGenerator
from seed import set_seed
from pathlib import Path

set_seed(42)

class DatasetGenerator:
    def __init__(
        self,
        processed_path: Path,
        video_path: Path,
        detector,
        task=2,
        test_size=0.3
    ):
        self.processed_path = Path(processed_path)
        self.video_path = Path(video_path)
        self.task = task
        self.test_size = test_size
        self.instance_generator = InstanceGenerator(detector)
        
        self.videos = self._get_videos()
        self.train_videos, self.test_videos = self._split_videos()
        
    def _get_videos(self):
        videos = [
            d.name for d in self.video_path.iterdir()
            if d.is_dir() and any(d.iterdir())
        ]
        return videos
    
    def _split_videos(self):
        train, test = train_test_split(self.videos, test_size=self.test_size)
        return train, test
    
    def generate_dataset(self, split="train", eye_gaze_data=None, save_metadata_path=None, sample_size=30):
        if split == "train":
            videos = self.train_videos
        elif split == "test":
            videos = self.test_videos
        else:
            raise ValueError(f"Invalid split '{split}'. Use 'train' or 'test'.")
        
        dataset = []
        
        for video in videos:
            frame_videos_path = self.video_path / video
            all_frames = sorted([f for f in frame_videos_path.iterdir() if f.is_file()])
            sampled_frames = random.sample(all_frames, min(sample_size, len(all_frames)))

            for frame_path in sampled_frames:
                if not frame_path.is_file():
                    continue
                frame = Image.open(frame_path)
                
                # Use instance generator to generate overlayed image and get label
                if self.task == 1 or self.task == 2:
                    generated_instance_data = self.instance_generator.generate(frame, frame_path, eye_gaze_data, self.task)
                    dataset.append({
                        "frames_file_names": [str(frame_path), str(generated_instance_data["save_path"])],
                        "label": generated_instance_data["label"]
                    })
                   
                elif self.task == 3:
                    generated_instance_1_data = self.instance_generator.generate(frame, frame_path, eye_gaze_data, self.task)
                    generated_instance_2_data = self.instance_generator.generate(frame, frame_path, eye_gaze_data, self.task)
                    instances = [generated_instance_1_data, generated_instance_2_data]
                    min_index = min(enumerate(instances), key=lambda x: x[1]["score"])[0] + 1

                    dataset.append({
                        "frames_file_names": [str(frame_path), str(generated_instance_1_data["save_path"]), str(generated_instance_2_data["save_path"])],
                        "label": min_index
                    })         
        
        if save_metadata_path:
            save_metadata_path = Path(save_metadata_path)
            save_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_metadata_path, "w") as f:
                for item in dataset:
                    f.write(json.dumps(item) + "\n")
            print(f"Saved dataset metadata to {save_metadata_path}")
        
        return dataset

if __name__ == "__main__":
   
    processed_path = Path("data") / "generated_overlays"
    video_path = Path("data") / "video_frames"
    eye_gaze_path = Path("data") / "eye_gaze_coords.csv"
    eye_gazes = pd.read_csv(eye_gaze_path)

    task_id = 3
    detector = SaliencyDetector()
    dataset_gen = DatasetGenerator(processed_path, video_path, detector, task=task_id)
    
    # Generate train dataset metadata and files
    dataset_gen.generate_dataset(
        split="train",
        eye_gaze_data=eye_gazes,
        save_metadata_path=f"data/train-task-{task_id}.jsonl"
    )

    # Uncomment to generate test set
    dataset_gen.generate_dataset(
        split="test",
        eye_gaze_data=eye_gazes,
        save_metadata_path=f"data/test-task-{task_id}.jsonl"
    ) 