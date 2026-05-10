import cv2
import numpy as np
import os
from shapely.geometry import Polygon

def detect_grid_squares(image_source, min_square_area=1000, max_square_area=10000000, debug=False):
    """
    Detects the Bürker chamber counting grid using a High-Pass Thin-Strip Template Matching algorithm.
    By using thin strips and a high-pass filter, it avoids rotation smear and perfectly isolates the grid lines 
    from dense pollen clusters.
    """
    if isinstance(image_source, str):
        img = cv2.imread(image_source)
        if img is None:
            raise ValueError(f"Could not read image: {image_source}")
    else:
        img = image_source.copy()
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    
    # Use a thin strip across the center to avoid rotation smear
    strip_w = min(1000, w)
    strip_h = min(1000, h)
    
    strip_x = edges[cy - strip_h//2 : cy + strip_h//2, :]
    strip_y = edges[:, cx - strip_w//2 : cx + strip_w//2]
    
    x_proj = np.sum(strip_x, axis=0)
    y_proj = np.sum(strip_y, axis=1)
    
    # High-pass filter: short smooth (keep lines) minus long smooth (remove pollen blobs)
    short_kernel = np.ones(5) / 5
    x_short = np.convolve(x_proj, short_kernel, mode='same')
    y_short = np.convolve(y_proj, short_kernel, mode='same')
    
    long_kernel = np.ones(101) / 101
    x_long = np.convolve(x_proj, long_kernel, mode='same')
    y_long = np.convolve(y_proj, long_kernel, mode='same')
    
    x_hp = x_short - x_long
    y_hp = y_short - y_long
    x_hp[x_hp < 0] = 0
    y_hp[y_hp < 0] = 0
    
    def get_best_grid(arr_x, arr_y):
        best_score = 0
        best_start_x = 0
        best_start_y = 0
        best_S = 187
        
        # A Bürker chamber has 4x4 squares in the corners.
        # So a corner has 5 lines: [0, 1, 2, 3, 4] * S
        # The central cross is 1mm wide, which is 4 * S.
        # So the next corner starts at 4 + 4 = 8.
        # The next corner has 5 lines: [8, 9, 10, 11, 12] * S
        template = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
        
        # S is typically ~180-210 pixels for a 5184x3456 image.
        scale = w / 3456.0
        min_S = max(10, int(150 * scale))
        max_S = max(20, int(250 * scale))
        
        for S in range(min_S, max_S):
            # Find best X
            best_x_score = 0
            best_x = 0
            for i in range(len(arr_x) - 12 * S):
                score = sum(arr_x[i + idx * S] for idx in template)
                if score > best_x_score:
                    best_x_score = score
                    best_x = i
                    
            # Find best Y
            best_y_score = 0
            best_y = 0
            for j in range(len(arr_y) - 12 * S):
                score = sum(arr_y[j + idx * S] for idx in template)
                if score > best_y_score:
                    best_y_score = score
                    best_y = j
                    
            # We want the highest COMBINED score where X and Y share the exact same spacing (perfect squares)
            combined = best_x_score + best_y_score
            if combined > best_score:
                best_score = combined
                best_start_x = best_x
                best_start_y = best_y
                best_S = S
                
        return best_start_x, best_start_y, best_S

    x_start, y_start, true_spacing = get_best_grid(x_hp, y_hp)
    
    polygons = []
    vis_img = img.copy() if debug else None
    
    # We have 4 corners. 
    # Top-Left: xi=0, yi=0
    # Top-Right: xi=8, yi=0
    # Bottom-Left: xi=0, yi=8
    # Bottom-Right: xi=8, yi=8
    
    for corner_xi in [0, 8]:
        for corner_yi in [0, 8]:
            corner_x = x_start + corner_xi * true_spacing
            corner_y = y_start + corner_yi * true_spacing
            
            # Each corner has a 4x4 grid of counting squares
            for i in range(4):
                for j in range(4):
                    sx1 = corner_x + i * true_spacing
                    sx2 = corner_x + (i + 1) * true_spacing
                    sy1 = corner_y + j * true_spacing
                    sy2 = corner_y + (j + 1) * true_spacing
                    
                    poly = Polygon([(sx1, sy1), (sx2, sy1), (sx2, sy2), (sx1, sy2)])
                    polygons.append(poly)
                    
                    if debug:
                        cv2.rectangle(vis_img, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (255, 0, 0), 4)

    if debug:
        if isinstance(image_source, str):
            base_name = os.path.basename(image_source)
            name, ext = os.path.splitext(base_name)
            out_path = os.path.join(os.path.dirname(image_source), f"{name}_grid{ext}")
        else:
            out_path = "debug_live_grid.jpg"
            
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
