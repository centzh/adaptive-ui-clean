"""
Saliency Detection Module for VLM

This module provides saliency detection methods for extracting saliency maps
for safety, aesthetics, and functionality from images. These maps can be
used as additional input features for VLM training.

Usage:
    from detectors import SaliencyDetector
    
    detector = SaliencyDetector()
    maps = detector.extract_all_maps(image_path, eye_gaze_data)
    detector.save_combined_map(maps, "output.png")
"""
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

class SaliencyDetector:
    def __init__(self,
                 ood_model_path: str = "yolo11n.pt",
                 seg_model_path: str = "nvidia/segformer-b5-finetuned-ade-640-640"):
        """
        Initialise the saliency detector.
        
        Args:
            ood_model_path: Path to YOLO model for object detection.
                Default is "yolo11n.pt".
            seg_model_path HuggingFace model for image segmentation.
                Default is "nvidia/segformer-b5-finetuned-ade-640-640".
        """
        self.ood_model = YOLO(ood_model_path)
        self.seg_model = pipeline("image-segmentation", model=seg_model_path)

        # Labels of objects that impact the safety of the setting. These are detected.
        self.safety_labels = [
            'floor', 'road', 'windowpane', 'person', 
            'door', 'signboard', 'stairs'
        ]

    def get_safety_map(self, frame: Image.Image):
        """
        Generate safety saliency map that highlights areas to avoid placing UI elements over
        """
        w, h = frame.size
        safety_map = np.zeros((h, w), dtype=bool)
        
        # Run segmentation
        segments = self.seg_model(frame)

        # Create the safety map by overlaying the masks of detected objects to avoid
        for segment in segments:
            if segment['label'] in self.safety_labels:
                mask = segment['mask'].resize((w, h))
                mask_np = np.array(mask, dtype=bool)
                safety_map |= mask_np

        # Convert to uint8
        safety_map = safety_map.astype(np.uint8) * 255
        return safety_map

    def get_aesthetics_map(self):
        pass

    def get_functionality_map(self):
        pass

    def get_all_maps(self):
        pass

    def save_map(self, saliency_map: np.ndarray, save_path: str):
        img = Image.fromarray(saliency_map)
        img.save(save_path)

    def combine_maps(self):
        pass

    def save_combined_map(self):
        pass

    def _extract_frame_id():
        pass

    def _find_closest_box(self):
        pass


# Example usage
if __name__ == "__main__":
    detector = SaliencyDetector()

    im_path = "frame-770.jpg"
    frame = Image.open(im_path)

    save_path = "result.png"
    safety_map = detector.get_safety_map(frame)
    detector.save_map(safety_map, save_path)