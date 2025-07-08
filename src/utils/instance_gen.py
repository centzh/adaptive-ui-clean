import numpy as np
import pandas as pd
import random
from PIL import Image, ImageDraw
from detectors import SaliencyDetector
from seed import set_seed
from pathlib import Path

set_seed(42)

class InstanceGenerator:
    """
    Generates overlay training instances from video frames and saliency maps.

    Parameters
    ----------
    detector : SaliencyDetector
        Object used to compute saliency or functionality maps.
    element_size : int, optional
        Size of the square overlay region (default is 400).
    step_size : int, optional
        Stride used when scanning across the image (default is 20).
    """
    def __init__(self, detector, element_height=319, element_width=663, step_size=20):
        self.detector = detector
        self.element_height = element_height
        self.element_width = element_width
        self.step_size = step_size
        self.scorer = ImageScorer(element_height, element_width, step_size)
        self.renderer = OverlayRenderer(element_height, element_width, step_size)

    def generate(self, frame: Image.Image, frame_path: str, eye_gazes: pd.DataFrame, task_id: int):
        """
        Generates and saves an overlay frame with the UI element on a selected region.

        Parameters
        ----------
        frame : PIL.Image.Image
            The input image frame.
        frame_path : str
            Path to the original frame image.
        eye_gazes : pandas.DataFrame
            DataFrame containing eye gaze coordinates.
        task_id : int
            Task identifier: 
            1 = visibility, functionality only, 
            2 = visibility, all factors, 
            3 = placement, all factors.
        """
        # Get combined saliency map
        if task_id == 1:
            saliency_map = self.detector.get_functionality_map(frame, frame_path, eye_gazes)
        else:
            saliency_map = self.detector.get_combined_map(frame, frame_path, eye_gazes)

        # Compute scores for the saliency map
        scores = self.scorer.get_scores(saliency_map)

        # Choose location
        sampler = LocationSampler(task_id)
        i, j, score, label = sampler.choose_location(scores)
        
        # Overlay and save
        frame_arr = np.array(frame)
        frame_arr = self.renderer.overlay(frame_arr, i, j)

        video_id = InstanceGenerator._get_video_id(frame_path)
        frame_id = InstanceGenerator._get_frame_id(frame_path)

        # Plot eye gaze points on the overlayed frame (returns a PIL Image)
        output_frame = self._plot_eye_gaze(frame_arr, eye_gazes, video_id, frame_id)

        if task_id == 1 or task_id == 2:
            save_name = f"frame-{frame_id}-{int(score)}-{label}.png"
        elif task_id == 3:
            save_name = f"frame-{frame_id}-{int(score)}.png"

        output_dir_path = f"data/generated_overlays/task_{task_id}/{video_id}"
        output_dir = Path(output_dir_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / save_name
        output_frame.save(save_path)

        return {"score":score, "label":label, "save_path": save_path}
    
    @staticmethod
    def _plot_eye_gaze(frame_arr: np.ndarray, eye_gazes: pd.DataFrame, video_id: str, frame_id: str):
        """
        Plot eye gaze points (from SaliencyDetector._get_eye_gaze_loc) on the image array using PIL.

        Parameters
        ----------
        frame_arr : np.ndarray
            The image array with UI element overlay.
        eye_gazes : pd.DataFrame
            The DataFrame containing eye gaze data.
        video_id : str
            Video identifier.
        frame_id : str
            Frame identifier.

        Returns
        -------
        PIL.Image.Image
            The image with gaze points plotted as small blue circles.
        """
        # Convert numpy array back to PIL Image for drawing
        img = Image.fromarray(frame_arr)
        draw = ImageDraw.Draw(img)

        # Get gaze x, y coordinates (assuming these are lists or arrays)
        gaze_x, gaze_y = SaliencyDetector._get_eye_gaze_loc(eye_gazes, video_id, frame_id)

        # Draw small blue circle for gaze point
        radius = 30
        left_up = (gaze_x - radius, gaze_y - radius)
        right_down = (gaze_x + radius, gaze_y + radius)
        draw.ellipse([left_up, right_down], fill=(255, 255, 0))

        return img

    @staticmethod
    def _get_video_id(frame_path: Path):
        return frame_path.parts[2]

    @staticmethod
    def _get_frame_id(frame_path: Path):
        return frame_path.stem.split("-")[1] 
        
class ImageScorer:
    """
    Computes average saliency scores across sliding windows on an image.

    Parameters
    ----------
    element_size : int
        Size of the window to compute scores over.
    step_size : int
        Stride used to move the window.
    """
    def __init__(self, element_height: int, element_width: int, step_size: int):
        self.element_height = element_height
        self.element_width = element_width
        self.step_size = step_size

    def get_scores(self, image: np.ndarray):
        """
        Computes saliency scores using mean pixel values in patches.

        Parameters
        ----------
        image : np.ndarray
            2D grayscale saliency or heatmap image.
        
        Returns
        -------
        np.ndarray
            A 2D array of average scores for each patch.
        """
        h, w = image.shape
        h_out = (h - self.element_height) // self.step_size + 1
        w_out = (w - self.element_width) // self.step_size + 1
        scores = np.zeros((h_out, w_out))

        for i in range(h_out):
            for j in range(w_out):
                top = i * self.step_size
                left = j * self.step_size
                patch = image[top:top+self.element_height, left:left+self.element_width]
                scores[i, j] = np.mean(patch)
        return scores
    
class LocationSampler:
    """
    Selects a location in the saliency score map based on task.

    Parameters
    ----------
    task_id : int
            Task identifier: 
            1 = visibility, functionality only, 
            2 = visibility, all factors, 
            3 = placement, all factors.
    """
    def __init__(self, task: int):
        self.task = task

    def choose_location(self, scores: np.ndarray):
        """
        Chooses a patch location either:
        1) Based on percentile thresholding -- Tasks 1, 2, or, 
        2) Randomly -- Task 3

        Parameters
        ----------
        scores : np.ndarray
            2D array of saliency scores.

        Returns
        -------
        tuple of (int, int, float, str or None)
            The row index, column index, selected score, and label (for tasks 1/2).
        """
        label = None
        if self.task == 1 or self.task == 2:
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
        i, j, score = random.choice(top_entries)
        return i, j, score, label
    

class OverlayRenderer:
    """
    Renders red overlays on image arrays at specified patch locations.

    Parameters
    ----------
    element_size : int
        Size of the square overlay region.
    step_size : int
        Stride used to locate overlay regions.
    """
    def __init__(self, element_height: int, element_width: int, step_size: int):
        self.element_height = element_height
        self.element_width = element_width
        self.step_size = step_size

        cwd = Path.cwd()  # Current working directory as a Path object
        light_path = cwd / "src" / "utils" / "ui_elements" / f"email-light.png"
        dark_path = cwd / "src" / "utils" / "ui_elements" / f"email-dark.png"
        self.light_overlay = Image.open(light_path).convert("RGBA")
        self.dark_overlay = Image.open(dark_path).convert("RGBA")

    def overlay(self, frame_arr: np.ndarray, i: int, j: int):
        """
        Overlays a red rectangle onto the image at (i, j) window index.

        Parameters
        ----------
        frame_arr : np.ndarray
            The original RGB image as a NumPy array.
        i : int
            Row index in the score map.
        j : int
            Column index in the score map.

        Returns
        -------
        np.ndarray
            The image with a red rectangle overlay.
        """
        # Convert to PIL Image with alpha channel
        frame_pil = Image.fromarray(frame_arr).convert("RGBA")

        # Compute position
        top = i * self.step_size
        left = j * self.step_size

        overlay_image = random.choice([self.light_overlay, self.dark_overlay])

        # Paste with mask to preserve transparency
        frame_pil.paste(overlay_image, (left, top), mask=overlay_image)

        # Convert back to numpy array
        return np.array(frame_pil.convert("RGBA"))
    
if __name__ == "__main__":
    detector = SaliencyDetector()
    frame_path = Path("data") / "video_frames" / "loc1_script1_seq1_rec1" / "frame-10.jpg"
    frame = Image.open(frame_path)
    eye_gaze_path = Path("data") / "eye_gaze_coords.csv"
    eye_gazes = pd.read_csv(eye_gaze_path)

    for task in range(1, 4):
        generator = InstanceGenerator(detector)
        generator.generate(frame, frame_path, eye_gazes, task)
