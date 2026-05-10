import cv2
import numpy as np
import sys
sys.path.append('.')
from src.line_detector import detect_grid_squares

squares = detect_grid_squares('data/raw_images/O.bi10_prov_dil1a10_2026_05_08_04.JPG', debug=True)
if len(squares) > 0:
    bounds = squares[0].bounds
    print(f"Square 0 bounds: {bounds}, width: {bounds[2]-bounds[0]}, height: {bounds[3]-bounds[1]}")
    
    # Calculate the spatial extent of all squares
    min_x = min([s.bounds[0] for s in squares])
    min_y = min([s.bounds[1] for s in squares])
    max_x = max([s.bounds[2] for s in squares])
    max_y = max([s.bounds[3] for s in squares])
    print(f"Grid extent: X({min_x} to {max_x}), Y({min_y} to {max_y})")
    
