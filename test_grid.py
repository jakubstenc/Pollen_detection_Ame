import cv2
import numpy as np

img = cv2.imread('data/raw_images/O.bi10_prov_dil1a10_2026_05_08_04.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adaptive thresholding instead of Canny might be better for faint lines
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)

# Morphological operations to extract horizontal and vertical lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

grid = cv2.add(horizontal_lines, vertical_lines)

# Find contours of the grid squares
contours, _ = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

vis = img.copy()
count = 0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    area = w * h
    # A 1mm square on 4000x3000 image is maybe 500x500 = 250,000 area.
    # A 1/16th subdivision is maybe 125x125 = 15,625 area.
    # Let's draw squares that are between 10,000 and 1,000,000
    if 10000 < area < 1000000:
        # Check if aspect ratio is roughly square
        if 0.8 < w/h < 1.2:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 4)
            count += 1

cv2.imwrite('data/results/grid_test.jpg', vis)
print(f"Found {count} squares using morphology.")
