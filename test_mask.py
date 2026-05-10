import cv2
import numpy as np

img = cv2.imread('data/raw_images/O.bi10_prov_dil1a10_2026_05_08_04_activemask.jpg', cv2.IMREAD_GRAYSCALE)
white_pixels = np.sum(img == 255)
total_pixels = img.size
print(f"Active mask coverage: {white_pixels / total_pixels * 100:.2f}%")

