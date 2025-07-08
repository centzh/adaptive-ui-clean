import numpy as np
import pandas as pd
import random
import os
from PIL import Image
from detectors import SaliencyDetector


seed = 42
random.seed(seed)
np.random.seed(seed)

def get_scores(image: np.ndarray, h: int, w: int, element_size: int, step_size: int):
    """
    Compute mean scores over sliding window patches of an image.

    Parameters
    ----------
    image: 2D input image (grayscale) as a NumPy array.
    h: Height of the input image.
    w: Width of the input image.
    element_size: Size of the square patch to extract (height and width).
    step_size: Step size between successive patches.

    Returns
    -------
    scores : 2D array of shape (h_out, w_out), where each entry is the mean 
    intensity of a patch extracted from the input image.
    """
    h_out = (h - element_size) // step_size + 1
    w_out = (w - element_size) // step_size + 1
    scores = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            top = i * step_size
            left = j * step_size
            patch = image[top:top+element_size, left:left+element_size]
            scores[i, j] = np.mean(patch)

    return scores

def overlay_frame(frame_arr, i, j, step_size, element_size):
    top = i * step_size
    left = j * step_size
    bottom = top + element_size
    right = left + element_size

    frame_arr[top:bottom, left:right, 0] = 255  # Red
    frame_arr[top:bottom, left:right, 1] = 0    # Green
    frame_arr[top:bottom, left:right, 2] = 0    # Blue
    return frame_arr

def choose_placement_location(task, scores, label=None):
    # Task 2: Visibility
    if task == 2:
        
        # Sample image label
        label = "yes" if random.random() < 0.5 else "no"
        # If label is no, sample from the top 95% of score locations
        # If label is yes, sample from the bottom 20% of score locations
        if label == "no":
            threshold = np.percentile(scores, 95)
            mask = scores >= threshold
        else:  # label == "yes"
            threshold = np.percentile(scores, 20)
            mask = scores <= threshold
        top_indices = np.argwhere(mask)
        print(label)

    # Task 3: Placement
    elif task == 3:
        top_indices = np.argwhere(np.ones_like(scores, dtype=bool))
    
    top_entries = [(i, j, scores[i, j]) for i, j in top_indices]
    i, j, _ = random.choice(top_entries)
    return i, j

def generate_instance(detector, frame, frame_path, eye_gazes, task, element_size=400, step_size=20):

    # Get saliency map
    combined_map = detector.get_combined_map(frame, frame_path, eye_gazes)
 
    # Get scores for the image
    frame_arr = np.array(frame)
    w, h = frame.size
    scores = get_scores(combined_map, h, w, element_size, step_size)

    # Choose element placement location
    i, j = choose_placement_location(task, scores)
     
    # Overlay location with element
    frame_arr = overlay_frame(frame_arr, i, j, step_size, element_size)
  
    overlaid_img = Image.fromarray(frame_arr)
    overlaid_img.save("result.png")
  
# Example usage
if __name__ == "__main__":
    detector = SaliencyDetector()
    frame_path = "data/video_frames/loc3_script1_seq7_rec1/frame-510.jpg"
    frame = Image.open(frame_path)
    eye_gaze_path = "data/eye_gaze_img_coords.csv"
    eye_gazes = pd.read_csv(eye_gaze_path)
    task = 2
    generate_instance(detector, frame, frame_path, eye_gazes, task)

