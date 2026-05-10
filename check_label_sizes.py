import glob
import os
files = glob.glob("data/dataset_yolo/train/labels/*.txt")
max_area = 0
for f in files:
    for line in open(f):
        parts = line.strip().split()
        if len(parts) > 4:
            coords = [float(p) for p in parts[1:]]
            xs = coords[0::2]
            ys = coords[1::2]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            area = w * h
            if area > max_area:
                max_area = area
                biggest_file = f
print(f"Max area: {max_area} in file {biggest_file}")
