import cv2
import numpy as np
from shapely.geometry import Polygon

def detect_grid_squares(image_path, min_square_area=10000, debug=False):
    """
    Detects the Bürker chamber grid lines and returns a list of Shapely Polygons
    representing the counting squares.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast to help line detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)
    
    if lines is None:
        print("No lines detected.")
        return []

    horizontals = []
    verticals = []
    
    # Separate lines into horizontal and vertical based on angle
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        
        if angle < 10 or angle > 170:
            horizontals.append(y1)
        elif 80 < angle < 100:
            verticals.append(x1)
            
    # Remove duplicates that are too close to each other
    def merge_nearby(lines_arr, threshold=20):
        if not lines_arr: return []
        lines_arr = sorted(lines_arr)
        merged = [lines_arr[0]]
        for val in lines_arr[1:]:
            if val - merged[-1] > threshold:
                merged.append(val)
        return merged
        
    horizontals = merge_nearby(horizontals)
    verticals = merge_nearby(verticals)
    
    # Form polygons from the grid intersections
    polygons = []
    for i in range(len(horizontals) - 1):
        for j in range(len(verticals) - 1):
            y1 = horizontals[i]
            y2 = horizontals[i+1]
            x1 = verticals[j]
            x2 = verticals[j+1]
            
            # Create a box
            area = abs((y2 - y1) * (x2 - x1))
            if area > min_square_area:
                # Top-left, Top-right, Bottom-right, Bottom-left
                poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                polygons.append(poly)
                
    if debug:
        # Draw the detected grid polygons on the image for debugging
        debug_img = img.copy()
        for poly in polygons:
            pts = np.array(poly.exterior.coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
        debug_path = image_path.replace(".jpg", "_grid.jpg").replace(".JPG", "_grid.jpg")
        cv2.imwrite(debug_path, debug_img)
        print(f"Debug grid image saved to: {debug_path}")
        
    return polygons

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        squares = detect_grid_squares(img_path, debug=True)
        print(f"Found {len(squares)} counting squares in {img_path}")
    else:
        print("Usage: python 03_line_detector.py <image_path>")
