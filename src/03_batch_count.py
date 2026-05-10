import os
import argparse
import glob
import csv
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from shapely.geometry import Polygon, Point
from line_detector import detect_grid_squares

def process_batch(image_dir, model_path, output_dir):
    """
    Processes all high-resolution images in the directory.
    Detects the Bürker grid, runs SAHI inference, applies the counting protocol,
    saves visualized images, and exports a CSV report.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "pollen_counts.csv")
    
    # Supported image extensions
    exts = ('*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG')
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
        
    print(f"Found {len(image_paths)} images to process.")
    print("Loading Detection Model...")
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.5,
        device="cpu", # Change to 'cuda:0' if GPU is available
    )
    
    results_data = []
    
    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        print(f"\n[{i+1}/{len(image_paths)}] Processing {filename}...")
        
        # 1. Grid Detection
        print("   - Detecting perfectly aligned mathematical grid...")
        try:
            grid_polygons = detect_grid_squares(img_path, debug=False)
            print(f"   - Found {len(grid_polygons)} counting squares.")
        except Exception as e:
            print(f"   - Error detecting grid: {e}")
            continue
            
        # 2. SAHI Inference
        print("   - Running Sliced Inference (this may take a minute)...")
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0
        )
        
        # 3. Protocol Enforcement
        print("   - Applying Bürker counting rules...")
        counted_pollen = []
        ignored_pollen = []
        
        for prediction in result.object_prediction_list:
            bbox = prediction.bbox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            pollen_center = Point(cx, cy)
            
            pollen_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            
            is_counted = False
            for grid_poly in grid_polygons:
                if grid_poly.contains(pollen_center):
                    # It's fully inside a square
                    is_counted = True
                    break
                elif grid_poly.intersects(pollen_poly):
                    # Intersects edge: Hemocytometer Rule
                    coords = list(grid_poly.exterior.coords)
                    top_line = Polygon([coords[0], coords[1], coords[1], coords[0]]).buffer(1)
                    right_line = Polygon([coords[1], coords[2], coords[2], coords[1]]).buffer(1)
                    bottom_line = Polygon([coords[2], coords[3], coords[3], coords[2]]).buffer(1)
                    left_line = Polygon([coords[3], coords[0], coords[0], coords[3]]).buffer(1)
                    
                    if pollen_poly.intersects(bottom_line) or pollen_poly.intersects(left_line):
                        is_counted = False
                        break
                    elif pollen_poly.intersects(top_line) or pollen_poly.intersects(right_line):
                        is_counted = True
                        break
                        
            if is_counted:
                counted_pollen.append(prediction)
            else:
                ignored_pollen.append(prediction)
                
        # 4. Visualization
        print("   - Generating Output Image...")
        img = cv2.imread(img_path)
        
        # Draw Blue Grid
        for poly in grid_polygons:
            pts = np.array(poly.exterior.coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
            
        # Draw Pollen
        def draw_predictions(preds, color):
            for p in preds:
                b = p.bbox
                cv2.rectangle(img, (int(b.minx), int(b.miny)), (int(b.maxx), int(b.maxy)), color, 3)
                cv2.putText(img, f"{p.score.value:.2f}", (int(b.minx), int(b.miny)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
                            
        draw_predictions(counted_pollen, (0, 255, 0)) # Green = Counted
        draw_predictions(ignored_pollen, (0, 0, 255)) # Red = Ignored
        
        out_name = f"visualized_{filename}"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, img)
        
        print(f"   - Saved: {out_name}")
        print(f"   - Pollen Counted: {len(counted_pollen)} | Ignored: {len(ignored_pollen)}")
        
        results_data.append({
            "Filename": filename,
            "Counted_Pollen": len(counted_pollen),
            "Ignored_Pollen": len(ignored_pollen),
            "Total_Detections": len(counted_pollen) + len(ignored_pollen)
        })
        
        # Save CSV progressively in case of crash
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Filename", "Counted_Pollen", "Ignored_Pollen", "Total_Detections"])
            writer.writeheader()
            writer.writerows(results_data)
            
    print(f"\n✅ Batch processing complete! Results saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process Bürker chamber images for pollen counting.")
    parser.add_argument("--image_dir", type=str, default="data/raw_images", help="Directory containing raw high-res images")
    parser.add_argument("--model", type=str, default="runs/pollen_nano_test-3/weights/best.pt", help="Path to YOLOv8 model weights")
    parser.add_argument("--output_dir", type=str, default="data/results", help="Directory to save visualized images and CSV")
    args = parser.parse_args()
    
    process_batch(args.image_dir, args.model, args.output_dir)
