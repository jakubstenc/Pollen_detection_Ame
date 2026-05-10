import os
import glob
import shutil
import datetime

def generate_report():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    runs_dir = os.path.join(project_root, "runs")
    docs_dir = os.path.join(project_root, "docs")
    assets_dir = os.path.join(docs_dir, "assets")
    
    os.makedirs(assets_dir, exist_ok=True)
    
    # Find the most recent training run directory
    run_folders = glob.glob(os.path.join(runs_dir, "pollen_nano_test*"))
    if not run_folders:
        print("No training runs found to generate a report.")
        return
        
    latest_run = max(run_folders, key=os.path.getmtime)
    print(f"Generating report from latest run: {os.path.basename(latest_run)}")
    
    # Count training images
    train_images_dir = os.path.join(project_root, "data", "dataset_yolo", "train", "images")
    num_images = 0
    if os.path.exists(train_images_dir):
        images = glob.glob(os.path.join(train_images_dir, "*.jpg")) + glob.glob(os.path.join(train_images_dir, "*.JPG"))
        num_images = len(images)
        
    # Files to copy to docs/assets
    target_files = {
        "results.png": "Model Training Metrics",
        "confusion_matrix.png": "Confusion Matrix",
        "train_batch0.jpg": "Ground Truth Training Data (What YOLO was taught)",
        "val_batch0_pred.jpg": "Validation Predictions (What YOLO guessed)"
    }
    
    copied_files = []
    for filename, title in target_files.items():
        src = os.path.join(latest_run, filename)
        if os.path.exists(src):
            dst = os.path.join(assets_dir, filename)
            shutil.copy2(src, dst)
            copied_files.append((filename, title))
            
    # Generate the Markdown file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content = f"# YOLOv8 Pollen Detection Model Performance\n\n"
    md_content += f"**Last Updated:** {timestamp}\n\n"
    md_content += f"**Training Images Used:** {num_images}\n\n"
    
    md_content += "This page automatically tracks the performance metrics of the latest YOLOv8 Nano instance segmentation training run.\n\n"
    
    for filename, title in copied_files:
        md_content += f"## {title}\n"
        md_content += f"![{title}](assets/{filename})\n\n"
        
    md_path = os.path.join(docs_dir, "index.md")
    with open(md_path, "w") as f:
        f.write(md_content)
        
    print(f"Successfully generated documentation at {md_path}")

if __name__ == "__main__":
    generate_report()
