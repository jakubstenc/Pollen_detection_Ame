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

def get_spacing(arr, min_dist):
    peaks = []
    threshold = np.mean(arr) + 0.5 * np.std(arr)
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > threshold:
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)
            elif arr[i] > arr[peaks[-1]]:
                peaks[-1] = i
    diffs = np.diff(peaks)
    valid_diffs = diffs[diffs > min_dist]
    if len(valid_diffs) == 0: return None
    return np.median(valid_diffs), peaks

min_distance = 100 # Should be smaller than 187
spacing_x, x_peaks = get_spacing(x_smooth, min_distance)
spacing_y, y_peaks = get_spacing(y_smooth, min_distance)

print(f"X Spacing: {spacing_x}")
print(f"X Peaks: {x_peaks}")
print(f"Y Spacing: {spacing_y}")
print(f"Y Peaks: {y_peaks}")
