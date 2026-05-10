import cv2
import numpy as np
import os
from shapely.geometry import Polygon

def detect_grid_squares(image_path, min_square_area=1000, max_square_area=10000000, debug=False):
    """
    Detects the Bürker chamber counting grid using a Pure Mathematical 1D Template Matching algorithm.
    The fundamental spacing S (~185px) is the size of the 0.25mm sub-squares.
    The 4 corners each contain a 4x4 grid of these sub-squares.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    x_proj = np.sum(edges, axis=0)
    y_proj = np.sum(edges, axis=1)
    
    kernel_size = max(11, int(img.shape[1] * (21.0 / 4000.0)))
    kernel = np.ones(kernel_size) / kernel_size
    x_smooth = np.convolve(x_proj, kernel, mode='same')
    y_smooth = np.convolve(y_proj, kernel, mode='same')
    
    def find_best_full_grid(arr):
        best_score = 0
        best_start = 0
        best_S = 187
        
        # A Bürker chamber has 4x4 squares in the corners.
        # So a corner has 5 lines: [0, 1, 2, 3, 4] * S
        # The central cross is 1mm wide, which is 4 * S.
        # So the next corner starts at 4 + 4 = 8.
        # The next corner has 5 lines: [8, 9, 10, 11, 12] * S
        template_indices = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
        
        # S is typically ~180-210 pixels for a 5184x3456 image.
        # Scale the search range based on image width to handle 640x640 tiles too!
        scale = img.shape[1] / 3456.0
        min_S = max(10, int(150 * scale))
        max_S = max(20, int(250 * scale))
        
        for S in range(min_S, max_S):
            for i in range(len(arr) - 12 * S):
                score = 0
                for idx in template_indices:
                    score += arr[i + idx * S]
                
                if score > best_score:
                    best_score = score
                    best_start = i
                    best_S = S
                    
        return best_start, best_S

    x_start, x_S = find_best_full_grid(x_smooth)
    y_start, y_S = find_best_full_grid(y_smooth)
    
    polygons = []
    vis_img = img.copy() if debug else None
    
    # We have 4 corners. 
    # Top-Left: xi=0, yi=0
    # Top-Right: xi=8, yi=0
    # Bottom-Left: xi=0, yi=8
    # Bottom-Right: xi=8, yi=8
    
    for corner_xi in [0, 8]:
        for corner_yi in [0, 8]:
            corner_x = x_start + corner_xi * x_S
            corner_y = y_start + corner_yi * y_S
            
            # Each corner has a 4x4 grid of counting squares
            for i in range(4):
                for j in range(4):
                    sx1 = corner_x + i * x_S
                    sx2 = corner_x + (i + 1) * x_S
                    sy1 = corner_y + j * y_S
                    sy2 = corner_y + (j + 1) * y_S
                    
                    poly = Polygon([(sx1, sy1), (sx2, sy1), (sx2, sy2), (sx1, sy2)])
                    polygons.append(poly)
                    
                    if debug:
                        cv2.rectangle(vis_img, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (255, 0, 0), 4)

    if debug:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        out_path = os.path.join(os.path.dirname(image_path), f"{name}_grid{ext}")
        cv2.imwrite(out_path, vis_img)
        print(f"Debug grid image saved to: {out_path}")
        
    return polygons

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        squares = detect_grid_squares(img_path, debug=True)
        print(f"Found {len(squares)} counting squares in {img_path}")
    else:
        print("Usage: python src/line_detector.py <image_path>")
