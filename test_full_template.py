import cv2
import numpy as np

img = cv2.imread('data/raw_images/O.bi10_prov_dil1a10_2026_05_08_04.JPG')
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

# We assume S is around 180 to 200 pixels.
def find_best_full_grid(arr):
    best_score = 0
    best_start = 0
    best_S = 187
    
    # Template lines relative to start: [0, 1, 2, 3, 4, 8, 9, 10, 11, 12] * S
    template_indices = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
    
    # Search over possible S values (170 to 210)
    for S in range(170, 210):
        # Search over possible start positions
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

print(f"X Start: {x_start}, X Spacing: {x_S}")
print(f"Y Start: {y_start}, Y Spacing: {y_S}")

# Calculate the 4 corner blocks
corner_blocks = []
for xi in [0, 8]:
    for yi in [0, 8]:
        corner_blocks.append((x_start + xi * x_S, y_start + yi * y_S))

print(f"Corner block starts (X, Y): {corner_blocks}")
