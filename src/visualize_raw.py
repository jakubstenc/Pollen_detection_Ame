import os
import argparse
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import importlib.util

sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in os.sys.path:
    os.sys.path.append(sys_path)
from src.line_detector import detect_grid_squares

def visualize_raw_image(image_path, model_path, output_dir):
    print(f"--- Processing High-Res Image: {os.path.basename(image_path)} ---")
    
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    print("1. Detecting 9-square mathematical counting grid on raw image...")
    macro_grids = detect_grid_squares(image_path, debug=False)
    total_squares = sum(len(g) for g in macro_grids.values())
    print(f"   Found {total_squares} counting squares across {len(macro_grids)} macro-regions.")

    print("2. Running SAHI Sliced Inference (this may take a minute on CPU)...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.3,
        device='cpu'
    )
    
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    
    print("3. Generating Visualization...")
    vis_img = img.copy()
    
    # Draw Grid (Thick Blue Lines)
    for label, grid_polygons in macro_grids.items():
        for poly in grid_polygons:
            pts = np.array(poly.exterior.coords, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts], isClosed=True, color=(255, 0, 0), thickness=8)
            
        if grid_polygons:
            b = grid_polygons[0].bounds
            cv2.putText(vis_img, label, (int(b[0]) + 40, int(b[1]) + 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 8)
        
    counted = 0
    ignored = 0
    
    # Process SAHI predictions
    for obj_pred in result.object_prediction_list:
        score = obj_pred.score.value
        bbox = obj_pred.bbox
        mask = obj_pred.mask
        
        # Centroid
        cx = bbox.minx + (bbox.maxx - bbox.minx) / 2
        cy = bbox.miny + (bbox.maxy - bbox.miny) / 2
        centroid = Point(cx, cy)
        
        from shapely.geometry import LineString
        
        pollen_box = Polygon([
            (bbox.minx, bbox.miny),
            (bbox.maxx, bbox.miny),
            (bbox.maxx, bbox.maxy),
            (bbox.minx, bbox.maxy)
        ])
        
        in_grid = False
        assigned_label = None
        
        for label, grid_polygons in macro_grids.items():
            for square in grid_polygons:
                sq_minx, sq_miny, sq_maxx, sq_maxy = square.bounds
                top_edge = LineString([(sq_minx, sq_miny), (sq_maxx, sq_miny)])
                bottom_edge = LineString([(sq_minx, sq_maxy), (sq_maxx, sq_maxy)])
                left_edge = LineString([(sq_minx, sq_miny), (sq_minx, sq_maxy)])
                right_edge = LineString([(sq_maxx, sq_miny), (sq_maxx, sq_maxy)])
                
                counted_in_this_square = False
                if pollen_box.intersects(bottom_edge) or pollen_box.intersects(left_edge):
                    counted_in_this_square = False
                elif pollen_box.intersects(top_edge) or pollen_box.intersects(right_edge):
                    counted_in_this_square = True
                elif square.contains(centroid):
                    counted_in_this_square = True
                    
                if counted_in_this_square:
                    in_grid = True
                    assigned_label = label
                    break
            if assigned_label is not None:
                break
                
        # Color: Green if counted, Red if ignored
        color = (0, 255, 0) if in_grid else (0, 0, 255)
        if in_grid: counted += 1
        else: ignored += 1
        
        # Draw Polygon Mask
        if mask is not None and mask.bool_mask is not None:
            contours, _ = cv2.findContours(mask.bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, color, 4)
            
        # Draw Centroid and Confidence
        cv2.circle(vis_img, (int(cx), int(cy)), 6, color, -1)
        cv2.putText(vis_img, f"{score:.2f}", (int(cx) + 10, int(cy) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    out_path = os.path.join(output_dir, f"visualized_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, vis_img)
    print(f"\n✅ Visualization saved to: {out_path}")
    print(f"📊 Summary: {counted} Pollen Counted (Green), {ignored} Ignored (Red)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Raw HD image path")
    parser.add_argument("--model", required=True, help="YOLO model path")
    parser.add_argument("--output", default="data/results", help="Output directory")
    args = parser.parse_args()
    visualize_raw_image(args.image, args.model, args.output)
