import os
import sys
import time

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from shapely.geometry import Point, Polygon, LineString
except ModuleNotFoundError as e:
    print("\n" + "="*80)
    print("🚨 ERROR: YOU CLICKED THE 'PLAY/RUN' BUTTON IN YOUR CODE EDITOR! 🚨")
    print("="*80)
    print(f"Missing module: {e}")
    print("\nYour code editor is ignoring our 'venv' virtual environment and using the system Python.")
    print("PLEASE DO NOT CLICK THE PLAY BUTTON!")
    print("\nInstead, open the Terminal window at the bottom of your screen and type exactly this:")
    print("python src/live_counter.py --camera 0 --model runs/pollen_nano_test-3/weights/best.pt")
    print("="*80 + "\n")
    sys.exit(1)

from line_detector import detect_grid_squares

def process_frame(frame, model, macro_grids):
    """
    Runs YOLO inference on a single frame and applies the counting protocol
    across all 9 macro squares.
    """
    results = model(frame, verbose=False)
    
    counted_pollen = []
    ignored_pollen = []
    region_counts = {label: 0 for label in macro_grids.keys()}
    
    # Draw Grid
    for label, grid_polygons in macro_grids.items():
        for poly in grid_polygons:
            pts = np.array(poly.exterior.coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            
        if grid_polygons:
            b = grid_polygons[0].bounds
            cv2.putText(frame, label, (int(b[0]) + 10, int(b[1]) + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
    for result in results:
        if result.masks is None:
            continue
            
        masks = result.masks.xy
        boxes = result.boxes
        scores = result.boxes.conf
        
        for mask_pts, box, score in zip(masks, boxes, scores):
            minx, miny, maxx, maxy = box.xyxy[0].tolist()
            cx = minx + (maxx - minx) / 2
            cy = miny + (maxy - miny) / 2
            centroid = Point(cx, cy)
            pollen_box = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
            
            is_counted = False
            assigned_label = None
            
            for label, grid_polygons in macro_grids.items():
                for square in grid_polygons:
                    sq_minx, sq_miny, sq_maxx, sq_maxy = square.bounds
                    top_edge = LineString([(sq_minx, sq_miny), (sq_maxx, sq_miny)])
                    bottom_edge = LineString([(sq_minx, sq_maxy), (sq_maxx, sq_maxy)])
                    left_edge = LineString([(sq_minx, sq_miny), (sq_minx, sq_maxy)])
                    right_edge = LineString([(sq_maxx, sq_miny), (sq_maxx, sq_maxy)])
                    
                    if pollen_box.intersects(bottom_edge) or pollen_box.intersects(right_edge):
                        is_counted = False
                        break
                    elif pollen_box.intersects(top_edge) or pollen_box.intersects(left_edge):
                        is_counted = True
                        assigned_label = label
                        break
                    elif square.contains(centroid):
                        is_counted = True
                        assigned_label = label
                        break
                        
                if assigned_label is not None:
                    break
                    
            if is_counted and assigned_label:
                counted_pollen.append((box, score.item()))
                region_counts[assigned_label] += 1
            else:
                ignored_pollen.append((box, score.item()))
                
    def draw_predictions(preds, color):
        for box, score in preds:
            minx, miny, maxx, maxy = box.xyxy[0].tolist()
            cv2.rectangle(frame, (int(minx), int(miny)), (int(maxx), int(maxy)), color, 2)

    draw_predictions(counted_pollen, (0, 255, 0)) # Green
    draw_predictions(ignored_pollen, (0, 0, 255)) # Red
    
    return frame, region_counts

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live Real-Time Pollen Counter (9-Square)")
    parser.add_argument("--camera", default="0", help="Camera index or RTSP stream URL")
    parser.add_argument("--model", required=True, help="Path to trained YOLO best.pt weights")
    args = parser.parse_args()

    print(f"Loading YOLO model from {args.model}...")
    model = YOLO(args.model)
    
    cam_id = int(args.camera) if args.camera.isdigit() else args.camera
    print(f"Opening camera stream: {cam_id}...")
    cap = cv2.VideoCapture(cam_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {cam_id}.")
        sys.exit(1)
        
    print("\n" + "="*50)
    print(" LIVE STREAM STARTED ")
    print(" Press 'q' to quit.")
    print(" Press 'g' to force re-detect the mathematical grid.")
    print("="*50 + "\n")
    
    macro_grids = {}
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
            
        if len(macro_grids) == 0 or frame_count % 60 == 0:
            try:
                macro_grids = detect_grid_squares(frame, debug=False)
            except Exception as e:
                pass
                
        start_time = time.time()
        annotated_frame, region_counts = process_frame(frame, model, macro_grids)
        fps = 1.0 / (time.time() - start_time)
        
        counts_list = list(region_counts.values()) if region_counts else []
        total_counted = sum(counts_list)
        avg = np.mean(counts_list) if counts_list else 0
        std = np.std(counts_list) if counts_list else 0
        
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Total Counted: {total_counted}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Avg/Sq: {avg:.1f}  StdDev: {std:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        display_frame = annotated_frame
        if annotated_frame.shape[1] > 1920:
            scale = 1920 / annotated_frame.shape[1]
            display_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)
            
        cv2.imshow("Live Pollen Counter (9-Square)", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            macro_grids = {}
            
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
