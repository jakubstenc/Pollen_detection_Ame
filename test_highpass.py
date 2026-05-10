import cv2
import numpy as np

img = cv2.imread('data/raw_images/O.bi10_prov_dil1a10_2026_05_08_04.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

x_proj = np.sum(edges, axis=0)
y_proj = np.sum(edges, axis=1)

# Short smoothing to reduce single-pixel noise but keep lines sharp
short_kernel = np.ones(5) / 5
x_short = np.convolve(x_proj, short_kernel, mode='same')
y_short = np.convolve(y_proj, short_kernel, mode='same')

# Long smoothing to estimate the background pollen mound
long_kernel = np.ones(101) / 101
x_long = np.convolve(x_proj, long_kernel, mode='same')
y_long = np.convolve(y_proj, long_kernel, mode='same')

# High-pass filter
x_hp = x_short - x_long
y_hp = y_short - y_long

# Set negative values to 0
x_hp[x_hp < 0] = 0
y_hp[y_hp < 0] = 0

def find_best_full_grid(arr):
    best_score = 0
    best_start = 0
    best_S = 187
    
    template_indices = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
    
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

x_start, x_S = find_best_full_grid(x_hp)
y_start, y_S = find_best_full_grid(y_hp)

print(f"X Start: {x_start}, X Spacing: {x_S}")
print(f"Y Start: {y_start}, Y Spacing: {y_S}")

