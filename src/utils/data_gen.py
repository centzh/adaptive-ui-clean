import numpy as np
import pandas as pd
import random
import os
from PIL import Image
from detectors import SaliencyDetector
from utils.seed import set_seed

set_seed(42)

class ImageScorer:
    def __init__(self, element_size: int, step_size: int):
        self.element_size = element_size
        self.step_size = step_size

    def get_scores(self, image: np.ndarray):
        h, w = image.shape
        h_out = (h - self.element_size) // self.step_size + 1
        w_out = (w - self.element_size) // self.step_size + 1
        scores = np.zeros((h_out, w_out))

        for i in range(h_out):
            for j in range(w_out):
                top = i * self.step_size
                left = j * self.step_size
                patch = image[top:top+self.element_size, left:left+self.element_size]
                scores[i, j] = np.mean(patch)
        return scores

class LocationSampler:
    def __init__(self, task: int):
        self.task = task

    def choose_location(self, scores: np.ndarray):
        label = None
        if self.task == 2:
            label = "yes" if random.random() < 0.5 else "no"
            if label == "no":
                threshold = np.percentile(scores, 95)
                mask = scores >= threshold
            else:
                threshold = np.percentile(scores, 20)
                mask = scores <= threshold
            top_indices = np.argwhere(mask)

        elif self.task == 3:
            top_indices = np.argwhere(np.ones_like(scores, dtype=bool))

        top_entries = [(i, j, scores[i, j]) for i, j in top_indices]
        i, j, _ = random.choice(top_entries)
        return i, j, label

class OverlayRenderer:
    def __init__(self, element_size: int, step_size: int):
        self.element_size = element_size
        self.step_size = step_size

    def overlay(self, frame_arr: np.ndarray, i: int, j: int):
        top = i * self.step_size
        left = j * self.step_size
        bottom = top + self.element_size
        right = left + self.element_size
        frame_arr[top:bottom, left:right, 0] = 255
        frame_arr[top:bottom, left:right, 1] = 0
        frame_arr[top:bottom, left:right, 2] = 0
        return frame_arr

class InstanceGenerator:
    def __init__(self, detector, element_size=400, step_size=20):
        self.detector = detector
        self.element_size = element_size
        self.step_size = step_size
        self.scorer = ImageScorer(element_size, step_size)
        self.renderer = OverlayRenderer(element_size, step_size)

    def generate(self, frame: Image.Image, frame_path: str, eye_gazes: pd.DataFrame, task: int, save_path="result.png"):
        # Get combined saliency map
        saliency_map = self.detector.get_combined_map(frame, frame_path, eye_gazes)

        # Compute scores for the saliency map
        scores = self.scorer.get_scores(saliency_map)

        # Choose location
        sampler = LocationSampler(task)
        i, j, label = sampler.choose_location(scores)
        print(f"Label: {label}")

        # Overlay and save
        frame_arr = np.array(frame)
        frame_arr = self.renderer.overlay(frame_arr, i, j)
        Image.fromarray(frame_arr).save(save_path)

if __name__ == "__main__":
    detector = SaliencyDetector()
    frame_path = "data/video_frames/loc3_script1_seq7_rec1/frame-510.jpg"
    frame = Image.open(frame_path)
    eye_gaze_path = "data/eye_gaze_img_coords.csv"
    eye_gazes = pd.read_csv(eye_gaze_path)
    task = 3
    generator = InstanceGenerator(detector)
    generator.generate(frame, frame_path, eye_gazes, task)
