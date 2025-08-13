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
import cv2
from pathlib import Path
import ipdb

MAP_COLORS = {
        'safety': (255, 99, 132),        # Soft red/pink
        'aesthetics': (54, 162, 235),    # Soft blue
        'functionality': (255, 206, 86), # Warm yellow
}
 
class SaliencyDetector:
    def __init__(self,
                 ood_model_path: str = "yolov8x-worldv2.pt",
                 seg_model_path: str = "nvidia/segformer-b5-finetuned-ade-640-640"
        ):
        """
        Initialise the saliency detector.
        
        Parameters
        ----------
        ood_model_path: Path to YOLO model for object detection.
            Default is "yolov8x-worldv2.pt".
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
        w, h = frame.size
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
        """
        Return a binary saliency map highlighting areas of high colour and edge intensity.

        Parameters
        ----------
        frame: Current video frame.

        Returns:
        --------
        sal_map: Binary saliency map with the same dimensions as the frame.
        """
        w, h = frame.size
        edge_map = SaliencyDetector._get_edges_map(frame)
        color_map = SaliencyDetector._get_color_map(frame)
        aesthetics_map = np.maximum(edge_map, color_map)
        return aesthetics_map
   
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
        
        # Extract bounding box coordinates for detected objects
        bboxes = pred.boxes
        obj_coords = bboxes.xyxy
        w, h = frame.size

        functionality_map = np.zeros((h, w), dtype=np.uint8)

        # No objects detected
        if obj_coords is None or len(obj_coords) == 0:
            return functionality_map # Return all zeros, if no object was found

        # Get current eye gaze location for the frame
        video_id = SaliencyDetector._get_video_id(frame_path)
        frame_id = SaliencyDetector._get_frame_id(frame_path)
        gaze_x, gaze_y = SaliencyDetector._get_eye_gaze_loc(eye_gazes, video_id, frame_id)

        if gaze_x is None and gaze_y is None:
            return functionality_map

        # Find object (midpoint) that is closest to the gaze point (1/4 of image width, by default)
        distance_thresh = w // 4 # CHANGE BACK to w//4
        closest_object = SaliencyDetector._find_closest_object(obj_coords, (gaze_x, gaze_y), distance_thresh)

        # Closest object was located at a distance greater than the threshold 
        if closest_object is None:
            return functionality_map # Return all zeros, if closest object was beyond threshold
        
        # Mark the bounding box of the closest object as white, representing to avoid
        x1, y1, x2, y2 = closest_object
        functionality_map[y1:y2, x1:x2] = 255
        return functionality_map
    
    def save_map(self, saliency_map: np.ndarray, save_path: str):
        img = Image.fromarray(saliency_map)
        img.save(save_path)
   
    def save_colored_map(self, sal_map: np.ndarray, save_path: str, color: tuple):
        """
        Save a binary/intensity saliency map as a transparent color overlay.

        Parameters
        ----------
        map_array : np.ndarray
            2D saliency map (binary or intensity).
        save_path : str
            Output path to save the image.
        color : tuple
            RGB color as (R, G, B).
        alpha : int
            Transparency value (0 to 255).
        """
        h, w = sal_map.shape

        if sal_map.dtype != np.uint8:
            norm_map = ((sal_map - sal_map.min()) / (sal_map.ptp() + 1e-8) * 255).astype(np.uint8)
        else:
            norm_map = sal_map

        gray_img = Image.fromarray(norm_map, mode='L')
        color_img = Image.new('RGBA', (w, h), color)   
        color_img.putalpha(gray_img)

        # Create black background (fully opaque)
        black_bg = Image.new('RGBA', (w, h), (0, 0, 0, 255))

        # Paste the colored image with alpha over black background
        black_bg.paste(color_img, (0, 0), color_img)

        # Save the final image with black background and colored overlay
        black_bg.save(save_path)


    def save_combined_map(self, maps: dict, save_path: str, use_color_overlay: bool = False):
        """
        Save a combined saliency map showing all maps.
        
        Parameters
        ----------
        maps : dict
            Dictionary of saliency maps. Keys are map names, values are 2D np.arrays (binary or grayscale).
        save_path : str
            Path to save the combined image.
        use_color_overlay : bool, optional (default False)
            If True, overlay each map in a distinct RGBA color with transparency.
            Otherwise, combine as grayscale by summing and clipping.
        """
        map_shape = next(iter(maps.values())).shape
        h, w = map_shape

        if use_color_overlay:
            combined_img = Image.new('RGBA', (w, h), (0, 0, 0, 255))

            for key, sal_map in maps.items():
                if key not in MAP_COLORS:
                    continue
                # Normalize map
                if sal_map.dtype != np.uint8:
                    norm_map = ((sal_map - sal_map.min()) / (sal_map.ptp() + 1e-8) * 255).astype(np.uint8)
                else:
                    norm_map = sal_map

                gray_img = Image.fromarray(norm_map, mode='L')
                color_img = Image.new('RGBA', (w, h), MAP_COLORS[key])
                color_img.putalpha(gray_img)

                combined_img = Image.alpha_composite(combined_img, color_img)

            combined_img.save(save_path)
        else:
            combined_map = np.zeros(map_shape, dtype=np.uint8)
            for sal_map in maps.values():
                if sal_map.dtype != np.uint8:
                    norm_map = ((sal_map - sal_map.min()) / (sal_map.ptp() + 1e-8) * 255).astype(np.uint8)
                else:
                    norm_map = sal_map
                combined_map = np.clip(combined_map + norm_map, 0, 255)

            combined_img = Image.fromarray(combined_map, mode='L')
            combined_img.save(save_path)

    def get_combined_map(self, frame, frame_path, eye_gazes):
        """
        Returns a binary saliency map representing areas that should be avoided, 
        based on all contextual factors: functionality, aesthetics, social acceptability and safety.

        Parameters
        ----------
        frame: Current video frame.
        frame_path: Path to the current video frame.
        eye_gazes: Dataframe containing the location of eyegazes recorded for each frame.

        Returns
        -------
        combined_map: Combined functionality, aesthetics, social acceptability and safety.
        """
        aesthetics_map = self.get_aesthetics_map(frame)
        safety_and_social_acceptability_map = self.get_safety_and_social_acceptability_map(frame)
        functionality_map = self.get_functionality_map(frame, frame_path, eye_gazes)
        combined_map = np.max(
            np.stack([aesthetics_map, safety_and_social_acceptability_map, functionality_map]),
            axis=0
        )
        return combined_map

    @staticmethod
    def _get_edges_map(frame: Image.Image):
        """
        Detect edges in the frame using Scharr operator
        """
        gray_frame = np.array(frame.convert("L"))

        # Get Scharr gradients
        grad_x = cv2.Scharr(gray_frame, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(gray_frame, cv2.CV_32F, 0, 1)

        # Get gradient magnitude
        edge_intensity = cv2.magnitude(grad_x, grad_y)
        
        # Normalise to 0-255 and convert to uint8
        edge_map = cv2.normalize(edge_intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return edge_map
    
    @staticmethod
    def _get_color_map(frame: Image.Image):
        """
        Extract the saturation channel from the frame as a grayscale saliency map
        """
        hsv_frame =  cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2HSV)
        color_map = hsv_frame[:, :, 1]
        return color_map
    
    @staticmethod
    def _get_video_id(frame_path: Path):
        return frame_path.parts[2]

    @staticmethod
    def _get_frame_id(frame_path: Path):
        return frame_path.stem.split("-")[1] 
        
    @staticmethod
    def _get_eye_gaze_loc(eye_gazes: pd.DataFrame, video_id: str, frame_id: str):
        video_frame = f"{video_id}_frame_{frame_id}"
        row = eye_gazes[eye_gazes["frame_id"]==video_frame]
        # No eye gazes found
        if row.empty:
            return None, None
        x, y = int(row['x'].values[0]), int(row['y'].values[0])
        return x, y

    @staticmethod
    def _find_closest_object(obj_coords, gaze_point, distance_thresh: float):
        def get_center(bbox):
            return ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
        
        def distance_to_gaze(bbox):
            center = get_center(bbox)
            return ((center[0] - gaze_point[0])**2 + (center[1] - gaze_point[1])**2)
        
        closest_bbox = min(obj_coords, key=distance_to_gaze)
        
        # Check if closest object is within threshold
        if distance_to_gaze(closest_bbox) > distance_thresh**2:  
            return None
        
        return [int(coord.item()) for coord in closest_bbox]
 

if __name__ == "__main__":
    detector = SaliencyDetector()

    eye_gaze_path = Path("data") / "eye_gaze_coords.csv"
    eye_gazes = pd.read_csv(eye_gaze_path)

    # List of frame paths to process
    frame_paths = [
        "data/video_frames/loc3_script2_seq7_rec1/frame-10.jpg",
        "data/video_frames/loc2_script1_seq5_rec1/frame-780.jpg",
        "data/video_frames/loc3_script3_seq4_rec2/frame-1320.jpg",
    ]

    for frame_path_str in frame_paths:
        frame_path = Path(frame_path_str)
        frame = Image.open(frame_path)

        # Create saliency maps
        functionality_map = detector.get_functionality_map(frame, frame_path, eye_gazes)
        aesthetics_map = detector.get_aesthetics_map(frame)
        safety_and_social_acceptability_map = detector.get_safety_and_social_acceptability_map(frame)
        combined_map = detector.get_combined_map(frame, frame_path, eye_gazes)

        # Generate a unique identifier for saving
        identifier = frame_path.parent.name + "_" + frame_path.stem

        # Save each color map
        detector.save_map(functionality_map, f"figures/{identifier}_functionality_binary.png")
        detector.save_map(aesthetics_map, f"figures/{identifier}_aesthetics_binary.png")
        detector.save_map(safety_and_social_acceptability_map, f"figures/{identifier}_safety_social_binary.png")

        # Save combined map with color overlay
        maps = {
            "safety": safety_and_social_acceptability_map,
            "aesthetics": aesthetics_map,
            "functionality": functionality_map
        }
        detector.save_map(combined_map, f"figures/{identifier}_combined.png")
