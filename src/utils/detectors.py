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
from PIL import Image
import pandas as pd
from ultralytics import YOLO
from transformers import pipeline
import ipdb
from PIL import ImageDraw
 
class SaliencyDetector:
    def __init__(self,
                 ood_model_path: str = "yolo11n.pt",
                 seg_model_path: str = "nvidia/segformer-b5-finetuned-ade-640-640"
        ):
        """
        Initialise the saliency detector.
        
        Parameters
        ----------
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

    def get_safety_and_social_acceptability_map(self, frame: Image.Image):
        """
        Return a binary saliency map that highlights areas of safety and social acceptability to avoid.
        This includes areas that jeopardise safety: floor, road, windowpane, door, signboard, stairs
        As well as areas that jeopardise social acceptability: persons.

        Parameters
        ----------
        frame: Current video frame.

        Returns
        -------
        sal_map: Binary saliency map with the same dimensions as the frame.
        """
        h, w = frame.size
        safety_social_map = np.zeros((h, w), dtype=bool)
        
        # Run segmentation on the frame
        segments = self.seg_model(frame)

        # Create the safety map by overlaying the masks of detected objects to avoid
        for segment in segments:
            if segment['label'] in self.safety_labels:
                mask = segment['mask'].resize((w, h))
                mask_np = np.array(mask, dtype=bool)
                safety_social_map |= mask_np

        safety_social_map = safety_social_map.astype(np.uint8) * 255
        return safety_social_map

    def get_aesthetics_map(self, frame: Image.Image):
        pass

    def get_functionality_map(self, 
                              frame: Image.Image, 
                              frame_path: str, 
                              eye_gazes: pd.DataFrame
        ):
        """
        Return a binary saliency map that represents areas of functionality that should be avoided.
        These are areas that indicate what the user may be currently or intending to do or interact with.
        
        Parameters
        ----------
        frame: Current video frame.
        frame_path: Path to the current video frame.
        eye_gazes: Dataframe containing the location of eyegazes recorded for each frame.

        Returns
        -------
        functionality_map: Binary saliency map with the same dimensions as the frame.
        """
        # Run object detection on the frame 
        pred = self.ood_model(frame)[0]
        pred.save(filename="prediction.jpg")
        
        # Extract bounding box coordinates for detected objects
        bboxes = pred.boxes
        obj_coords = bboxes.xyxy
        w, h = frame.size

        # No objects detected
        if obj_coords is None or len(obj_coords) == 0:
            return None

        # Get current eye gaze location for the frame
        video_id = SaliencyDetector._get_video_id(frame_path)
        frame_id = SaliencyDetector._get_frame_id(frame_path)
        gaze_x, gaze_y = SaliencyDetector._get_eye_gaze_loc(eye_gazes, video_id, frame_id)

                
        # Draw gaze point on frame
        draw = ImageDraw.Draw(frame)
        draw.ellipse(
            [(gaze_x - 5, gaze_y - 5), (gaze_x + 5, gaze_y + 5)],
            fill='red'
        )

        # Save image with gaze point
        frame.save("prediction_with_gaze.jpg")

        # Find object (midpoint) that is closest to the gaze point (1/4 of image width, by default)
        distance_thresh = w//4
        closest_object = SaliencyDetector._find_closest_object(obj_coords, (gaze_x, gaze_y), distance_thresh)

        # Closest object was located at a distance greater than the threshold 
        if closest_object is None:
            return None
        
        # Mark the bounding box of the closest object as white, representing to avoid
        x1, y1, x2, y2 = closest_object
        functionality_map = np.zeros((h, w), dtype=np.uint8)
        functionality_map[y1:y2, x1:x2] = 255
        return functionality_map

    def get_all_maps(self):
        pass

    def save_map(self, saliency_map: np.ndarray, save_path: str):
        img = Image.fromarray(saliency_map)
        img.save(save_path)

    def combine_maps(self):
        pass

    def save_combined_map(self):
        pass
    
    @staticmethod
    def _get_video_id(frame_path: str):
        video_id = frame_path.split("/")[2]
        return video_id
    
    @staticmethod
    def _get_frame_id(frame_path: str):
        frame_id = frame_path.split("/")[3].split(".jpg")[0].split("-")[1]
        return frame_id
    
    @staticmethod
    def _get_eye_gaze_loc(eye_gazes: pd.DataFrame, video_id: str, frame_id: str):
        video_frame = f"{video_id}_frame_{frame_id}"
        row = eye_gazes[eye_gazes["frame_id"]==video_frame]
        x, y = int(row['x'].values[0]), int(row['y'].values[0])
        return x, y

    @staticmethod
    def _find_closest_object(obj_coords, gaze_point, distance_thresh: float):
        closest_point = min(obj_coords, key=lambda p: ((p[0] + p[2])/2 - gaze_point[0])**2 + ((p[1]+p[3])/2 - gaze_point[1])**2)
        closest_point = [int(coord.item()) for coord in closest_point]
        if (closest_point[0] - gaze_point[0])**2 + (closest_point[1] - gaze_point[1])**2 > distance_thresh:
            return None
        return closest_point
 
    

# Example usage
if __name__ == "__main__":
    detector = SaliencyDetector()

    frame_path = "data/video_frames/loc3_script1_seq5_rec1/frame-660.jpg"
    frame = Image.open(frame_path)

    # save_path = "safety_social_acceptability_map.png"
    # safety_map = detector.get_safety_and_social_acceptability_map(frame)
    # detector.save_map(safety_map, save_path)

    save_path = "functionality_map.png"
    eye_gaze_path = "data/eye_gaze_img_coords.csv"
    eye_gazes = pd.read_csv(eye_gaze_path)
    safety_map = detector.get_functionality_map(frame, frame_path, eye_gazes)
    detector.save_map(safety_map, save_path)