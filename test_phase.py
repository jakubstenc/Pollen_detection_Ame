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

def find_peaks(arr, min_dist):
    peaks = []
    threshold = np.mean(arr) + 1.5 * np.std(arr) # HIGH THRESHOLD
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > threshold:
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)
            elif arr[i] > arr[peaks[-1]]:
                peaks[-1] = i
    return peaks

min_distance = int(img.shape[1] * (40.0 / 4000.0))
x_peaks = find_peaks(x_smooth, min_distance)
y_peaks = find_peaks(y_smooth, min_distance)

spacings = [s for s in [spacing_x, spacing_y] if s is not None]
min_s = min(spacings)
if min_s > 150:
    true_spacing = min_s / 4.0
elif min_s > 75:
    true_spacing = min_s / 2.0
else:
    true_spacing = min_s

print(f"X Spacing: {spacing_x}")
print(f"Y Spacing: {spacing_y}")
print(f"True Sub-grid Spacing: {true_spacing}")
