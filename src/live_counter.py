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

# PIXEL_TO_UM_RATIO represents how many micrometers are in 1 pixel.
PIXEL_TO_UM_RATIO = 1.0 

def process_frame(frame, model, grid_polygons):
    """
    Runs YOLO inference on a single frame and applies the counting protocol.
    Returns the annotated frame and the counts.
    """
    # 1. Run YOLO Instance Segmentation (Raw inference, no SAHI)
    results = model(frame, verbose=False)
    
    counted_pollen = []
    ignored_pollen = []
    
    # Draw Grid
    for poly in grid_polygons:
        pts = np.array(poly.exterior.coords, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        
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
            
            pollen_box = Polygon([
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy)
            ])
            
            is_counted = False
            for square in grid_polygons:
                sq_minx, sq_miny, sq_maxx, sq_maxy = square.bounds
                
                top_edge = LineString([(sq_minx, sq_miny), (sq_maxx, sq_miny)])
                bottom_edge = LineString([(sq_minx, sq_maxy), (sq_maxx, sq_maxy)])
                left_edge = LineString([(sq_minx, sq_miny), (sq_minx, sq_maxy)])
                right_edge = LineString([(sq_maxx, sq_miny), (sq_maxx, sq_maxy)])
                
                # Hemocytometer logic
                if pollen_box.intersects(bottom_edge) or pollen_box.intersects(left_edge):
                    is_counted = False
                    break
                elif pollen_box.intersects(top_edge) or pollen_box.intersects(right_edge):
                    is_counted = True
                    break
                elif square.contains(centroid):
                    is_counted = True
                    break
                    
            if is_counted:
                counted_pollen.append((box, score.item()))
            else:
                ignored_pollen.append((box, score.item()))
                
    # Draw bounding boxes
    def draw_predictions(preds, color):
        for box, score in preds:
            minx, miny, maxx, maxy = box.xyxy[0].tolist()
            cv2.rectangle(frame, (int(minx), int(miny)), (int(maxx), int(maxy)), color, 2)
            cv2.putText(frame, f"{score:.2f}", (int(minx), int(miny)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    draw_predictions(counted_pollen, (0, 255, 0)) # Green = Counted
    draw_predictions(ignored_pollen, (0, 0, 255)) # Red = Ignored
    
    return frame, len(counted_pollen), len(ignored_pollen)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live Real-Time Pollen Counter")
    parser.add_argument("--camera", default="0", help="Camera index (e.g., 0) or RTSP stream URL")
    parser.add_argument("--model", required=True, help="Path to trained YOLO best.pt weights")
    args = parser.parse_args()

    # Load YOLO Model
    print(f"Loading YOLO model from {args.model}...")
    model = YOLO(args.model)
    
    # Initialize Camera
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
    
    grid_polygons = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
            
        # Detect grid automatically on the very first frame, or every 60 frames to track movement
        if len(grid_polygons) == 0 or frame_count % 60 == 0:
            try:
                # The mathematical template matcher is fast, so we can run it dynamically
                grid_polygons = detect_grid_squares(frame, debug=False)
            except Exception as e:
                pass # If grid fails to detect (e.g. out of focus), keep the old one or empty list
                
        # Run YOLO inference
        start_time = time.time()
        annotated_frame, counted, ignored = process_frame(frame, model, grid_polygons)
        fps = 1.0 / (time.time() - start_time)
        
        # Add Overlay text
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Counted: {counted}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Ignored: {ignored}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Resize window if frame is too huge (e.g. 5184x3456)
        display_frame = annotated_frame
        if annotated_frame.shape[1] > 1920:
            scale = 1920 / annotated_frame.shape[1]
            display_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)
            
        cv2.imshow("Live Pollen Counter (Press 'q' to quit, 'g' to redraw grid)", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            grid_polygons = [] # Force redetect on next loop
            
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
