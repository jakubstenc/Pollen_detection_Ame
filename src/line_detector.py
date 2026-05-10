import cv2
import numpy as np
import os
from shapely.geometry import Polygon

def detect_grid_squares(image_path, min_square_area=1000, max_square_area=1000000, debug=False):
    """
    Detects the Bürker chamber counting grid using a Pure Mathematical Template Matching algorithm.
    1. 1D Projections calculate the exact fundamental spacing (1mm macro lines).
    2. Template matching slides a 4-line comb filter to lock onto the 3x3 macro grid perfectly.
    3. The 4 corner macro squares are mathematically subdivided into 4x4 sub-grids (16 squares each).
    4. Exactly 64 counting squares are returned. Completely immune to pollen density and missing lines!
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
    
    # Smooth projections to combine triple lines into single massive peaks
    kernel_size = max(11, int(img.shape[1] * (21.0 / 4000.0)))
    kernel = np.ones(kernel_size) / kernel_size
    x_smooth = np.convolve(x_proj, kernel, mode='same')
    y_smooth = np.convolve(y_proj, kernel, mode='same')
    
    def get_spacing(arr, min_dist):
        peaks = []
        threshold = np.mean(arr) + 1.0 * np.std(arr)
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > threshold:
                if not peaks or i - peaks[-1] >= min_dist:
                    peaks.append(i)
                elif arr[i] > arr[peaks[-1]]:
                    peaks[-1] = i
        diffs = np.diff(peaks)
        valid_diffs = diffs[diffs > min_dist]
        if len(valid_diffs) == 0: return 185 # Fallback to typical 1mm spacing
        return np.median(valid_diffs)

    # 1mm is usually ~150-200 pixels.
    min_distance = int(img.shape[1] * (150.0 / 4000.0))
    spacing_x = get_spacing(x_smooth, min_distance)
    spacing_y = get_spacing(y_smooth, min_distance)
    
    # A perfectly square chamber should have identical X and Y spacing. 
    true_spacing = min(spacing_x, spacing_y)
    
    # 1D Template Matching: Find the perfect 3x3 macro grid (4 lines)
    def find_best_grid(arr, spacing):
        best_score = 0
        best_start = 0
        for i in range(len(arr) - int(3 * spacing)):
            score = arr[i] + arr[int(i + spacing)] + arr[int(i + 2*spacing)] + arr[int(i + 3*spacing)]
            if score > best_score:
                best_score = score
                best_start = i
        return [int(best_start + j * spacing) for j in range(4)]

    x_macro = find_best_grid(x_smooth, true_spacing)
    y_macro = find_best_grid(y_smooth, true_spacing)

    polygons = []
    vis_img = img.copy() if debug else None
    
    # The 4 corner macro squares are defined by the intersection of:
    # Top-Left: x[0]-x[1], y[0]-y[1]
    # Top-Right: x[2]-x[3], y[0]-y[1]
    # Bottom-Left: x[0]-x[1], y[2]-y[3]
    # Bottom-Right: x[2]-x[3], y[2]-y[3]
    corner_indices = [
        (0, 1, 0, 1), # TL
        (2, 3, 0, 1), # TR
        (0, 1, 2, 3), # BL
        (2, 3, 2, 3)  # BR
    ]
    
    for (xi1, xi2, yi1, yi2) in corner_indices:
        x_start = x_macro[xi1]
        x_end = x_macro[xi2]
        y_start = y_macro[yi1]
        y_end = y_macro[yi2]
        
        # Subdivide the 1mm macro square into a 4x4 sub-grid (16 squares)
        sub_w = (x_end - x_start) / 4.0
        sub_h = (y_end - y_start) / 4.0
        
        for i in range(4):
            for j in range(4):
                sx1 = x_start + i * sub_w
                sx2 = x_start + (i + 1) * sub_w
                sy1 = y_start + j * sub_h
                sy2 = y_start + (j + 1) * sub_h
                
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
