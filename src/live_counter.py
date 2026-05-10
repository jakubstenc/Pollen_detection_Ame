import os
import sys
import cv2
import pandas as pd
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import importlib.util

from line_detector import detect_grid_squares

# PIXEL_TO_UM_RATIO represents how many micrometers are in 1 pixel.
# Change this value once you calibrate your Canon microscope setup!
PIXEL_TO_UM_RATIO = 1.0 

def estimate_volume(pixel_area, ratio=PIXEL_TO_UM_RATIO):
    """
    Estimates the physical volume of a pollen grain assuming it is roughly spherical.
    V = (4/3) * pi * r^3. Area = pi * r^2.
    """
    import math
    physical_area = pixel_area * (ratio ** 2)
    radius = math.sqrt(physical_area / math.pi)
    volume = (4.0 / 3.0) * math.pi * (radius ** 3)
    return volume

def process_image(image_path, model_path, output_dir):
    print(f"--- Processing {image_path} ---")
    
    # 1. Detect Grid
    grid_polygons = detect_grid_squares(image_path, debug=True)
    if not grid_polygons:
        print("Warning: No Bürker grid squares detected. Bypassing counting.")
        return []
        
    print(f"Detected {len(grid_polygons)} grid squares.")
    
    # 2. Run YOLO Instance Segmentation
    model = YOLO(model_path)
    results = model(image_path, verbose=False)
    
    pollen_data = []
    
    # 3. Spatial Join
    for result in results:
        if result.masks is None:
            continue
            
        masks = result.masks.xy # Coordinates of polygons
        boxes = result.boxes    # Bounding boxes
        
        for mask_pts, box in zip(masks, boxes):
            # Calculate centroid of the bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centroid = Point(cx, cy)
            
            # Check if this centroid falls inside ANY of the detected grid squares
            in_grid = False
            for square in grid_polygons:
                if square.contains(centroid):
                    in_grid = True
                    break
            
            if in_grid:
                # Calculate pixel area of the mask contour
                import numpy as np
                contour = np.array(mask_pts, dtype=np.int32)
                pixel_area = cv2.contourArea(contour)
                volume = estimate_volume(pixel_area)
                
                pollen_data.append({
                    "image": os.path.basename(image_path),
                    "pixel_area": pixel_area,
                    "estimated_volume_um3": volume
                })
                
    print(f"Found {len(pollen_data)} pollen grains inside the counting grid!")
    return pollen_data

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pollen Counter Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", required=True, help="Path to trained YOLO best.pt weights")
    parser.add_argument("--output", default="output.csv", help="Output CSV path")
    args = parser.parse_args()

    results = process_image(args.image, args.model, os.path.dirname(args.output))
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")
        print(df.describe())
    else:
        print("\nNo pollen counted. Output CSV not created.")

if __name__ == "__main__":
    main()
