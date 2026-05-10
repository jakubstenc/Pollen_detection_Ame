import cv2
import numpy as np

img = cv2.imread('data/raw_images/O.bi10_prov_dil1a10_2026_05_08_04.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Enhance contrast using CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# Adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)

# Morphological operations to extract horizontal and vertical lines
kernel_length = np.array(img).shape[1]//150
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

# A slight closing to bridge small gaps
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

# Thicken the lines to ensure intersections are solid
horizontal_lines = cv2.dilate(horizontal_lines, np.ones((3,3), np.uint8), iterations=2)
vertical_lines = cv2.dilate(vertical_lines, np.ones((3,3), np.uint8), iterations=2)

grid = cv2.add(horizontal_lines, vertical_lines)

# Find contours of the grid squares
contours, _ = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

vis = img.copy()
count = 0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    area = w * h
    
    # Filter for the 0.25mm sub-squares (roughly 5000 to 100,000 pixels)
    # AND filter for the 1mm massive squares (up to 500,000 pixels)
    if 5000 < area < 1000000:
        # Must be roughly square
        if 0.7 < w/h < 1.3:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 4)
            count += 1

cv2.imwrite('data/results/grid_test2.jpg', vis)
print(f"Found {count} sub-squares using morphological grid extraction.")
