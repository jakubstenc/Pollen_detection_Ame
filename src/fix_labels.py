import os
import glob
import math

def adapt_square_to_circle(coords_list, num_points=16):
    """
    Takes a flat list of normalized coordinates [x1, y1, x2, y2, ...]
    forming a polygon (e.g. a square).
    Returns a flat list of coordinates forming a circular polygon
    (e.g., a hexadecagon) inscribed within the bounding box of the original.
    """
    # Group into (x, y) pairs
    points = [(coords_list[i], coords_list[i+1]) for i in range(0, len(coords_list), 2)]
    
    # Calculate bounding box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Calculate centroid and radius
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    
    width = x_max - x_min
    height = y_max - y_min
    
    # We want a circle that fits tightly inside the bounding box
    # If the user's AI assist drew imperfect rectangles, min() guarantees it won't bleed outside.
    radius = min(width, height) / 2.0
    
    # Generate points around the circle
    circle_coords = []
    for i in range(num_points):
        theta = 2.0 * math.pi * i / num_points
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        
        # Clamp to [0, 1] just in case of floating point weirdness
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
        circle_coords.extend([x, y])
        
    return circle_coords

def process_labels(labels_dir):
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if not label_files:
        print(f"No label files found in {labels_dir}")
        return
        
    for file_path in label_files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue # Skip invalid lines
                
            class_id = parts[0]
            coords = [float(p) for p in parts[1:]]
            
            # Generate the new circular polygon
            new_coords = adapt_square_to_circle(coords, num_points=16)
            
            # Format back to string
            new_line = f"{class_id} " + " ".join([f"{c:.6f}" for c in new_coords]) + "\n"
            new_lines.append(new_line)
            
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
            
    print(f"Successfully adapted {len(label_files)} label files in {labels_dir}")

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_labels = os.path.join(project_root, "data", "dataset_yolo", "train", "labels")
    val_labels = os.path.join(project_root, "data", "dataset_yolo", "valid", "labels")
    
    if os.path.exists(train_labels):
        process_labels(train_labels)
    
    if os.path.exists(val_labels):
        process_labels(val_labels)
        
if __name__ == "__main__":
    main()
