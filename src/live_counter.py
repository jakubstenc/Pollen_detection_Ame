import os
import sys

try:
    import cv2
    import pandas as pd
    from ultralytics import YOLO
    from shapely.geometry import Point, Polygon
except ModuleNotFoundError as e:
    print("\n" + "="*80)
    print("🚨 ERROR: YOU CLICKED THE 'PLAY/RUN' BUTTON IN YOUR CODE EDITOR! 🚨")
    print("="*80)
    print(f"Missing module: {e}")
    print("\nYour code editor is ignoring our 'venv' virtual environment and using the system Python.")
    print("PLEASE DO NOT CLICK THE PLAY BUTTON!")
    print("\nInstead, open the Terminal window at the bottom of your screen and type exactly this:")
    print("python src/live_counter.py --image data/tiles_640/your_tile_name.jpg --model runs/pollen_nano_test-3/weights/best.pt")
    print("="*80 + "\n")
    sys.exit(1)
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
        
        from shapely.geometry import LineString
        
        for mask_pts, box in zip(masks, boxes):
            # Calculate bounds of the bounding box
            minx, miny, maxx, maxy = box.xyxy[0].tolist()
            cx = minx + (maxx - minx) / 2
            cy = miny + (maxy - miny) / 2
            centroid = Point(cx, cy)
            
            pollen_box = Polygon([
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy)
            ])
            
            in_grid = False
            for square in grid_polygons:
                sq_minx, sq_miny, sq_maxx, sq_maxy = square.bounds
                
                # Define the 4 geometric edges of the counting square
                top_edge = LineString([(sq_minx, sq_miny), (sq_maxx, sq_miny)])
                bottom_edge = LineString([(sq_minx, sq_maxy), (sq_maxx, sq_maxy)])
                left_edge = LineString([(sq_minx, sq_miny), (sq_minx, sq_maxy)])
                right_edge = LineString([(sq_maxx, sq_miny), (sq_maxx, sq_maxy)])
                
                # Hemocytometer counting protocol:
                counted_in_this_square = False
                if pollen_box.intersects(bottom_edge) or pollen_box.intersects(left_edge):
                    counted_in_this_square = False  # Touching bottom/left -> DO NOT COUNT
                elif pollen_box.intersects(top_edge) or pollen_box.intersects(right_edge):
                    counted_in_this_square = True   # Touching top/right -> DO COUNT
                elif square.contains(centroid):
                    counted_in_this_square = True   # Strictly inside -> DO COUNT
                    
                if counted_in_this_square:
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
