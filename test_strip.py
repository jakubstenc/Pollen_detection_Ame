import cv2
import numpy as np

img = cv2.imread('data/raw_images/O.bi10_prov_dil1a10_2026_05_08_04.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

h, w = img.shape[:2]
cx, cy = w // 2, h // 2

# Use a 500-pixel wide strip across the center to avoid rotation smear
strip_h = 500
strip_w = 500
strip_x = edges[cy - strip_h//2 : cy + strip_h//2, :]
strip_y = edges[:, cx - strip_w//2 : cx + strip_w//2]

x_proj = np.sum(strip_x, axis=0)
y_proj = np.sum(strip_y, axis=1)

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
    
    template = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
    
    scale = w / 3456.0
    min_S = max(10, int(160 * scale))
    max_S = max(20, int(220 * scale))
    
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
                
        # Combined score
        combined = best_x_score + best_y_score
        if combined > best_score:
            best_score = combined
            best_start_x = best_x
            best_start_y = best_y
            best_S = S
            
    return best_start_x, best_start_y, best_S

x_start, y_start, S = get_best_grid(x_hp, y_hp)
print(f"X Start: {x_start}, Y Start: {y_start}, Spacing: {S}")

