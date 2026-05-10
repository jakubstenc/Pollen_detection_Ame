import glob
import numpy as np
files = glob.glob("data/dataset_yolo/train/labels/*.txt")
areas = []
for f in files:
    for line in open(f):
        parts = line.strip().split()
        if len(parts) > 4:
            coords = [float(p) for p in parts[1:]]
            xs = coords[0::2]
            ys = coords[1::2]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            areas.append(w * h)

areas = np.array(areas)
print(f"Total annotations: {len(areas)}")
print(f"Median area: {np.median(areas)}")
print(f"90th percentile: {np.percentile(areas, 90)}")
print(f"99th percentile: {np.percentile(areas, 99)}")
print(f"Max area: {np.max(areas)}")

# How many are above 0.05?
big_ones = np.sum(areas > 0.02)
print(f"Number of annotations covering > 2% of image: {big_ones}")
