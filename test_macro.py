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

# Find the 4 strongest peaks!
def find_top_peaks(arr, num_peaks, min_dist):
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            peaks.append((i, arr[i]))
    
    # Sort by amplitude
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    filtered = []
    for p, amp in peaks:
        if not any(abs(p - fp) < min_dist for fp in filtered):
            filtered.append(p)
            if len(filtered) == num_peaks:
                break
    return sorted(filtered)

# 1mm is ~185 pixels.
min_distance = int(img.shape[1] * (150.0 / 4000.0))
x_macro = find_top_peaks(x_smooth, 4, min_distance)
y_macro = find_top_peaks(y_smooth, 4, min_distance)

print(f"X Macro Lines: {x_macro}")
print(f"Y Macro Lines: {y_macro}")

