import glob
import os

def clean_large_labels():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_labels = glob.glob(os.path.join(project_root, "data", "dataset_yolo", "train", "labels", "*.txt"))
    val_labels = glob.glob(os.path.join(project_root, "data", "dataset_yolo", "valid", "labels", "*.txt"))
    
    all_files = train_labels + val_labels
    removed_count = 0
    
    for f in all_files:
        if not os.path.exists(f): continue
        with open(f, 'r') as file:
            lines = file.readlines()
            
        valid_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 4:
                coords = [float(p) for p in parts[1:]]
                xs = coords[0::2]
                ys = coords[1::2]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                area = w * h
                
                # A normal pollen grain covers ~0.1% of the image. 
                # If it's > 1%, it's almost certainly a bad AI hallucination box.
                if area <= 0.01:
                    valid_lines.append(line)
                else:
                    removed_count += 1
                    
        with open(f, 'w') as file:
            file.writelines(valid_lines)
            
    print(f"Cleaned up {removed_count} massive hallucinated labels from the dataset!")

if __name__ == "__main__":
    clean_large_labels()
