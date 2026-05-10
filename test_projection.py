import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data/raw_images/O.bi10_prov_dil1a10_2026_05_08_04.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# Edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 1D Projections (summing the edge pixels along the axes)
# To avoid rotation issues, we can compute this on small overlapping horizontal/vertical strips,
# or assume the grid is mostly axis-aligned (which it seems to be).
x_proj = np.sum(edges, axis=0)
y_proj = np.sum(edges, axis=1)

# Smooth the projections to combine the triple lines into a single peak
kernel_size = 31 # Roughly the width of the triple lines
kernel = np.ones(kernel_size) / kernel_size
x_smooth = np.convolve(x_proj, kernel, mode='same')
y_smooth = np.convolve(y_proj, kernel, mode='same')

# Find peaks
def find_peaks(arr, distance):
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            # Threshold to ignore noise
            if arr[i] > np.mean(arr) + 1.5 * np.std(arr):
                if not peaks or i - peaks[-1] >= distance:
                    peaks.append(i)
                elif arr[i] > arr[peaks[-1]]:
                    # Replace with higher peak if too close
                    peaks[-1] = i
    return peaks

# Expected distance between grid lines
# On a 4000x3000 image, 0.25mm lines are maybe 100-200 pixels apart
min_distance = int(img.shape[1] * (50.0 / 4000.0))

x_peaks = find_peaks(x_smooth, min_distance)
y_peaks = find_peaks(y_smooth, min_distance)

# Draw the mathematical grid
vis = img.copy()
for x in x_peaks:
    cv2.line(vis, (x, 0), (x, img.shape[0]), (255, 0, 0), 2)
for y in y_peaks:
    cv2.line(vis, (0, y), (img.shape[1], y), (0, 255, 0), 2)

# Save visualization
cv2.imwrite('data/results/projection_grid.jpg', vis)
print(f"Found {len(x_peaks)} vertical lines and {len(y_peaks)} horizontal lines.")
