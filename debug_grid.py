import cv2
import numpy as np
from shapely.geometry import Polygon

img_path = 'data/raw_images/O.bi10_prov_dil1a10_2026_05_08_04.JPG'
img = cv2.imread(img_path)
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

def find_peaks(arr, min_dist):
    peaks = []
    threshold = np.mean(arr) + 1.0 * np.std(arr)
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > threshold:
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)
            elif arr[i] > arr[peaks[-1]]:
                peaks[-1] = i
    return peaks

min_distance = int(img.shape[1] * (50.0 / 4000.0))
x_peaks = find_peaks(x_smooth, min_distance)
y_peaks = find_peaks(y_smooth, min_distance)

print(f"min_dist: {min_distance}")
print(f"x_peaks: {x_peaks}")
print(f"y_peaks: {y_peaks}")

def interpolate_grid(peaks, min_dist):
    if len(peaks) < 2:
        return peaks
    diffs = np.diff(peaks)
    valid_diffs = diffs[diffs > min_dist]
    if len(valid_diffs) == 0:
        return peaks
        
    spacing = np.median(valid_diffs)
    print(f"spacing: {spacing}")
    math_peaks = []
    current = float(peaks[0])
    while current <= peaks[-1] + spacing/2:
        math_peaks.append(int(current))
        current += spacing
    return math_peaks

x_math = interpolate_grid(x_peaks, min_distance)
y_math = interpolate_grid(y_peaks, min_distance)

print(f"x_math: {x_math}")
print(f"y_math: {y_math}")

