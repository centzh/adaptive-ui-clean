import numpy as np
import pandas as pd
import random
import os
from PIL import Image
from detectors import SaliencyDetector



def


# Example usage
if __name__ == "__main__":
    detector = SaliencyDetector()

    frame_path = "data/video_frames/loc3_script1_seq7_rec1/frame-510.jpg"
    frame = Image.open(frame_path)
    eye_gaze_path = "data/eye_gaze_img_coords.csv"
    eye_gazes = pd.read_csv(eye_gaze_path)
    combined_map = detector.get_combined_map(frame, frame_path, eye_gazes)
    detector.save_map(combined_map, "combined.png")