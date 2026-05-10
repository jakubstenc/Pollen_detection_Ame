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

# We know the macro spacing is ~185 pixels.
# Let's dynamically find it by finding the fundamental frequency of the top peaks.
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
    if len(valid_diffs) == 0: return 185
    return np.median(valid_diffs)

min_distance = int(img.shape[1] * (150.0 / 4000.0))
spacing_x = get_spacing(x_smooth, min_distance)
spacing_y = get_spacing(y_smooth, min_distance)
true_spacing = min(spacing_x, spacing_y)

# Template matching: A template of 4 lines, separated by true_spacing.
def find_best_grid(arr, spacing):
    best_score = 0
    best_start = 0
    # Slide the template across the image
    for i in range(len(arr) - int(3 * spacing)):
        # Score is the sum of the signal at the 4 line positions
        score = arr[i] + arr[int(i + spacing)] + arr[int(i + 2*spacing)] + arr[int(i + 3*spacing)]
        if score > best_score:
            best_score = score
            best_start = i
    return [int(best_start + j * spacing) for j in range(4)]

x_macro = find_best_grid(x_smooth, true_spacing)
y_macro = find_best_grid(y_smooth, true_spacing)

print(f"X Macro: {x_macro}")
print(f"Y Macro: {y_macro}")
